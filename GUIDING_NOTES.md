# Path Guiding Notes

This repository contains stubbed support for CPU trained / GPU sampled path guiding.
The implementation follows the OpenPGL based interface but does not yet provide a
fully functional guiding solution. Use the `--guiding` command line flag together
with the `guiding` cargo feature to enable the scaffolding.

Currently only placeholder logic is present; OpenPGL data is ignored and sampling
falls back to the original behaviour.
