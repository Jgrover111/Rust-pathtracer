# Path Guiding Notes

Experimental path guiding support is available and uses CPU training with GPU sampling via an OpenPGL-style interface.

## Enabling guiding

Guiding is off by default. Build with the `guiding` cargo feature and launch with the `--guiding` flag:

```bash
cargo run --release --features guiding -- --guiding
```

Both the feature and flag are required; without them the renderer falls back to standard BSDF sampling.

### Optional flags

- `--guide-update-interval=N` – upload a new guide to the GPU every `N` batches (default `1`).
- `--guide-grid=XxYxZ` or `--guide-grid=N` – resolution of the guiding grid; a single value applies to all axes.

## Update cadence

The guiding state accumulates training data continuously. A snapshot is pushed to the GPU after every batch specified by `--guide-update-interval`. Larger intervals reduce overhead but delay adaptation.

## Scope

Guiding currently covers surface interactions only. Volume scattering and light-source guiding are not yet implemented and use standard sampling.

## Performance expectations

This prototype adds CPU work for training and extra GPU memory traffic. With frequent updates the overhead can outweigh quality gains; expect slower renders compared to unguided runs and benchmark on your own hardware.

## Known limitations

- Guide statistics reset when the application exits; no persistence.
- Only surface scattering is guided.
- Falls back to unbiased sampling when no guide data is available.
- OpenPGL data formats and APIs may change.

Future updates will refine the implementation and expand guiding coverage.
