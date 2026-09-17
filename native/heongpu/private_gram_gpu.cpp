// HFL-M3 -- https://github.com/kt4ngw/HFL-M3
// Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
// GPU BFV Gram-matrix backend for the one-time virtual-set preprocessing.
//
// This executable is a protocol-emulation harness: client histograms are
// encrypted, the plaintext buffer is released, and the evaluator receives
// only deserialized ciphertexts.  The selector decrypts only pairwise inner
// products.  Production deployments should place these roles in separate
// processes and transport HEonGPU's serialized objects between them.

#include <heongpu/heongpu.hpp>

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr auto Scheme = heongpu::Scheme::BFV;
constexpr char kInputMagic[8] = {'H', 'F', 'L', 'M', 'G', 'R', 'M', '1'};
constexpr char kOutputMagic[8] = {'H', 'F', 'L', 'M', 'G', 'M', 'T', '1'};
constexpr std::uint32_t kFormatVersion = 1;

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T> T read_scalar(std::ifstream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream)
        throw std::runtime_error("Truncated private-Gram input file");
    return value;
}

template <typename T> void write_scalar(std::ofstream& stream, T value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!stream)
        throw std::runtime_error("Could not write private-Gram output file");
}

struct InputData
{
    std::uint32_t clients = 0;
    std::uint32_t classes = 0;
    std::vector<std::uint64_t> histograms;
};

InputData read_input(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open input file: " + path);

    char magic[8]{};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kInputMagic, sizeof(magic)) != 0)
        throw std::runtime_error("Invalid private-Gram input magic");

    const auto version = read_scalar<std::uint32_t>(stream);
    if (version != kFormatVersion)
        throw std::runtime_error("Unsupported private-Gram input version");

    InputData input;
    input.clients = read_scalar<std::uint32_t>(stream);
    input.classes = read_scalar<std::uint32_t>(stream);
    if (input.clients == 0 || input.classes == 0)
        throw std::runtime_error("Client and class counts must be positive");

    const std::uint64_t element_count =
        static_cast<std::uint64_t>(input.clients) * input.classes;
    if (element_count > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Histogram matrix is too large");
    input.histograms.resize(static_cast<std::size_t>(element_count));
    stream.read(reinterpret_cast<char*>(input.histograms.data()),
                static_cast<std::streamsize>(element_count * sizeof(std::uint64_t)));
    if (!stream)
        throw std::runtime_error("Truncated histogram matrix");
    return input;
}

void write_output(const std::string& path, std::uint32_t clients,
                  const std::vector<std::uint64_t>& gram)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("Cannot open output file: " + path);
    stream.write(kOutputMagic, sizeof(kOutputMagic));
    write_scalar(stream, kFormatVersion);
    write_scalar(stream, clients);
    stream.write(reinterpret_cast<const char*>(gram.data()),
                 static_cast<std::streamsize>(gram.size() * sizeof(std::uint64_t)));
    if (!stream)
        throw std::runtime_error("Could not finish private-Gram output file");
}

std::vector<int> reduction_shifts(std::uint32_t classes,
                                  std::size_t poly_modulus_degree)
{
    const std::size_t row_size = poly_modulus_degree / 2;
    if (classes > row_size)
        throw std::runtime_error(
            "The number of classes exceeds one BFV batching row");
    std::vector<int> shifts;
    for (std::uint32_t shift = 1; shift < classes; shift <<= 1)
        shifts.push_back(static_cast<int>(shift));
    return shifts;
}

struct Pair
{
    std::uint32_t first;
    std::uint32_t second;
};

std::vector<Pair> upper_triangle_pairs(std::uint32_t clients)
{
    const std::uint64_t count =
        static_cast<std::uint64_t>(clients) * (clients + 1) / 2;
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<std::size_t>(count));
    for (std::uint32_t i = 0; i < clients; ++i)
        for (std::uint32_t j = i; j < clients; ++j)
            pairs.push_back({i, j});
    return pairs;
}

void evaluate_batch(
    heongpu::HEContext<Scheme>& context,
    heongpu::HEArithmeticOperator<Scheme>& operators,
    heongpu::Relinkey<Scheme>& relin_key,
    heongpu::Galoiskey<Scheme>& galois_key,
    const std::vector<int>& shifts,
    std::vector<heongpu::Ciphertext<Scheme>>& encrypted_histograms,
    const std::vector<Pair>& pairs, std::size_t begin, std::size_t count,
    const std::vector<cudaStream_t>& streams,
    std::vector<heongpu::Ciphertext<Scheme>>& outputs)
{
    const int thread_count = static_cast<int>(streams.size());

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::int64_t offset = 0; offset < static_cast<std::int64_t>(count);
         ++offset)
    {
        const int thread_id = omp_get_thread_num();
        const auto options =
            heongpu::ExecutionOptions().set_stream(streams[thread_id]);
        const Pair pair = pairs[begin + static_cast<std::size_t>(offset)];

        heongpu::Ciphertext<Scheme> product(context, options);
        operators.multiply(encrypted_histograms[pair.first],
                           encrypted_histograms[pair.second], product, options);
        operators.relinearize_inplace(product, relin_key, options);

        // A power-of-two rotate-and-add reduction puts the dot product in
        // slot zero. Slots between C and the next power of two are zero.
        for (const int shift : shifts)
        {
            heongpu::Ciphertext<Scheme> rotated(context, options);
            operators.rotate_rows(product, rotated, galois_key, shift, options);
            operators.add_inplace(product, rotated, options);
        }
        outputs[static_cast<std::size_t>(offset)] = std::move(product);
    }
}

void decrypt_batch(
    heongpu::HEContext<Scheme>& context,
    heongpu::Secretkey<Scheme>& secret_key,
    std::vector<heongpu::Ciphertext<Scheme>>& encrypted_values,
    std::size_t count, const std::vector<cudaStream_t>& streams,
    std::vector<std::uint64_t>& values)
{
    const int thread_count = static_cast<int>(streams.size());

#pragma omp parallel num_threads(thread_count)
    {
        const int thread_id = omp_get_thread_num();
        const auto options =
            heongpu::ExecutionOptions().set_stream(streams[thread_id]);
        heongpu::HEEncoder<Scheme> encoder(context);
        heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);

#pragma omp for schedule(static)
        for (std::int64_t offset = 0;
             offset < static_cast<std::int64_t>(count); ++offset)
        {
            heongpu::Plaintext<Scheme> plaintext(context, options);
            decryptor.decrypt(plaintext,
                              encrypted_values[static_cast<std::size_t>(offset)],
                              options);
            std::vector<std::uint64_t> decoded;
            encoder.decode(decoded, plaintext, options);
            values[static_cast<std::size_t>(offset)] = decoded.at(0);
        }
    }
}

std::size_t parse_positive_size(const char* value, const char* name)
{
    const auto parsed = std::stoull(value);
    if (parsed == 0)
        throw std::runtime_error(std::string(name) + " must be positive");
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3 || argc > 7)
        {
            std::cerr
                << "Usage: " << argv[0]
                << " INPUT.bin OUTPUT.bin [poly_degree=8192]"
                   " [plain_modulus=33832961] [streams=16] [batch=128]\n";
            return 2;
        }

        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const std::size_t poly_modulus_degree =
            argc >= 4 ? parse_positive_size(argv[3], "poly_degree") : 8192;
        const std::uint64_t plain_modulus =
            argc >= 5 ? parse_positive_size(argv[4], "plain_modulus") : 33832961;
        const std::size_t stream_count =
            argc >= 6 ? parse_positive_size(argv[5], "streams") : 16;
        const std::size_t batch_size =
            argc >= 7 ? parse_positive_size(argv[6], "batch") : 128;

        auto input = read_input(input_path);
        if (stream_count > 128)
            throw std::runtime_error("streams must not exceed 128");
        auto shifts = reduction_shifts(input.classes, poly_modulus_degree);

        std::uint64_t max_samples = 0;
        for (std::uint32_t client = 0; client < input.clients; ++client)
        {
            std::uint64_t samples = 0;
            for (std::uint32_t label = 0; label < input.classes; ++label)
                samples += input.histograms[
                    static_cast<std::size_t>(client) * input.classes + label];
            max_samples = std::max(max_samples, samples);
        }
        if (max_samples != 0 &&
            max_samples > (plain_modulus - 1) / max_samples)
            throw std::runtime_error(
                "plain_modulus is too small for the public dot-product bound");

        const auto total_start = Clock::now();
        auto stage_start = Clock::now();

        heongpu::HEContext<Scheme> context(
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus_default_values(1);
        context.set_plain_modulus(static_cast<int>(plain_modulus));
        context.generate();

        heongpu::HEKeyGenerator<Scheme> keygen(context);
        heongpu::Secretkey<Scheme> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Publickey<Scheme> public_key(context);
        keygen.generate_public_key(public_key, secret_key);
        heongpu::Relinkey<Scheme> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);
        heongpu::Galoiskey<Scheme> galois_key(context, shifts);
        keygen.generate_galois_key(galois_key, secret_key);

        const auto context_buffer = heongpu::serializer::serialize(context);
        const auto public_key_buffer =
            heongpu::serializer::serialize(public_key);
        const auto relin_key_buffer = heongpu::serializer::serialize(relin_key);
        const auto galois_key_buffer =
            heongpu::serializer::serialize(galois_key);
        const double keygen_seconds = seconds_since(stage_start);

        stage_start = Clock::now();
        heongpu::HEEncoder<Scheme> client_encoder(context);
        heongpu::HEEncryptor<Scheme> client_encryptor(context, public_key);
        std::vector<std::vector<std::uint8_t>> encrypted_packets;
        encrypted_packets.reserve(input.clients);
        std::uint64_t encrypted_histogram_bytes = 0;
        for (std::uint32_t client = 0; client < input.clients; ++client)
        {
            std::vector<std::uint64_t> message(poly_modulus_degree, 0);
            for (std::uint32_t label = 0; label < input.classes; ++label)
                message[label] = input.histograms[
                    static_cast<std::size_t>(client) * input.classes + label];
            heongpu::Plaintext<Scheme> plaintext(context);
            client_encoder.encode(plaintext, message);
            heongpu::Ciphertext<Scheme> ciphertext(context);
            client_encryptor.encrypt(ciphertext, plaintext);
            encrypted_packets.push_back(
                heongpu::serializer::serialize(ciphertext));
            encrypted_histogram_bytes += encrypted_packets.back().size();
        }
        const double client_encryption_seconds = seconds_since(stage_start);

        // Erase the only plaintext copy before entering the evaluator stage.
        std::fill(input.histograms.begin(), input.histograms.end(), 0);
        input.histograms.clear();
        input.histograms.shrink_to_fit();

        stage_start = Clock::now();
        std::vector<heongpu::Ciphertext<Scheme>> encrypted_histograms;
        encrypted_histograms.reserve(input.clients);
        for (const auto& packet : encrypted_packets)
            encrypted_histograms.push_back(
                heongpu::serializer::deserialize<
                    heongpu::Ciphertext<Scheme>>(packet));
        const double evaluator_input_load_seconds = seconds_since(stage_start);

        std::vector<cudaStream_t> streams(stream_count);
        for (auto& stream : streams)
            check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");

        heongpu::HEEncoder<Scheme> evaluator_encoder(context);
        heongpu::HEArithmeticOperator<Scheme> operators(context,
                                                        evaluator_encoder);
        const auto pairs = upper_triangle_pairs(input.clients);
        std::vector<std::uint64_t> gram(
            static_cast<std::size_t>(input.clients) * input.clients, 0);
        double evaluation_seconds = 0.0;
        double decryption_seconds = 0.0;

        for (std::size_t begin = 0; begin < pairs.size(); begin += batch_size)
        {
            const std::size_t count =
                std::min(batch_size, pairs.size() - begin);
            std::vector<heongpu::Ciphertext<Scheme>> encrypted_values(count);

            stage_start = Clock::now();
            evaluate_batch(context, operators, relin_key, galois_key, shifts,
                           encrypted_histograms, pairs, begin, count, streams,
                           encrypted_values);
            for (const auto stream : streams)
                check_cuda(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize after evaluation");
            evaluation_seconds += seconds_since(stage_start);

            stage_start = Clock::now();
            std::vector<std::uint64_t> values(count);
            decrypt_batch(context, secret_key, encrypted_values, count, streams,
                          values);
            for (const auto stream : streams)
                check_cuda(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize after decryption");
            decryption_seconds += seconds_since(stage_start);

            for (std::size_t offset = 0; offset < count; ++offset)
            {
                const Pair pair = pairs[begin + offset];
                const auto value = values[offset] % plain_modulus;
                gram[static_cast<std::size_t>(pair.first) * input.clients +
                     pair.second] = value;
                gram[static_cast<std::size_t>(pair.second) * input.clients +
                     pair.first] = value;
            }
        }

        for (auto stream : streams)
            check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        write_output(output_path, input.clients, gram);
        const double total_seconds = seconds_since(total_start);
        const std::uint64_t client_context_bytes =
            context_buffer.size() + public_key_buffer.size();
        const std::uint64_t evaluator_context_bytes =
            client_context_bytes + relin_key_buffer.size() +
            galois_key_buffer.size();

        std::cout << std::fixed << std::setprecision(6)
                  << "HFLM_GPU_JSON {"
                  << "\"backend\":\"HEonGPU-BFV\","
                  << "\"clients\":" << input.clients << ','
                  << "\"classes\":" << input.classes << ','
                  << "\"poly_modulus_degree\":" << poly_modulus_degree << ','
                  << "\"plain_modulus\":" << plain_modulus << ','
                  << "\"cuda_streams\":" << stream_count << ','
                  << "\"batch_size\":" << batch_size << ','
                  << "\"encrypted_gram_entries\":" << pairs.size() << ','
                  << "\"keygen_seconds\":" << keygen_seconds << ','
                  << "\"client_encryption_and_serialization_seconds\":"
                  << client_encryption_seconds << ','
                  << "\"evaluator_input_deserialization_seconds\":"
                  << evaluator_input_load_seconds << ','
                  << "\"gram_evaluation_seconds\":" << evaluation_seconds
                  << ','
                  << "\"gram_decryption_seconds\":" << decryption_seconds
                  << ','
                  << "\"gram_evaluation_and_decryption_seconds\":"
                  << evaluation_seconds + decryption_seconds << ','
                  << "\"total_gpu_protocol_seconds\":" << total_seconds << ','
                  << "\"client_context_bytes\":" << client_context_bytes
                  << ','
                  << "\"evaluator_context_bytes\":"
                  << evaluator_context_bytes << ','
                  << "\"encrypted_histogram_bytes\":"
                  << encrypted_histogram_bytes << ','
                  << "\"role_emulation\":true}"
                  << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "private_gram_gpu: " << error.what() << std::endl;
        return 1;
    }
}
