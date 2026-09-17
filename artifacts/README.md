# Generated artifacts

This directory separates generated experiment state from source code.

- `virtual_groups/groups/`: cached virtual-group membership produced by
  `scripts/prepare_private_virtual_groups.py`, together with a
  `.privacy.json` sidecar recording the BFV preprocessing runtime.

`virtual_groups/` holds the Fashion-MNIST example cache used by the README
quick start. Do not manually edit the pickle files; rebuild them with
`scripts/prepare_private_virtual_groups.py --force`.
