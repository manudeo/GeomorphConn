# CLI Usage

## Basic run

```bash
geomorphconn run \
    --dem dem.tif \
    --ndvi ndvi.tif \
    --rainfall rainfall.tif \
    --weight-factors rainfall ndvi roughness \
    --weight-combine mean \
    --flow-director DINF \
    --reference-grid dem \
    --use-aspect-weighting \
    --outputs IC Dup Ddn \
    --out-dir outputs
```

## Useful commands

```bash
geomorphconn welcome
geomorphconn --version
```

## Target mode (vector)

```bash
geomorphconn run \
    --dem dem.tif \
    --ndvi ndvi.tif \
    --rainfall rainfall.tif \
    --target-vector river.shp \
    --all-touched \
    --target-buffer 5 \
    --outputs IC
```

For complete option details, see [options.md](options.md).
