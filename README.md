## Setup Instructions

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Working-Length Structure

```text
src/working_length/
|-- annotation/       # manual ROI and tooth-mask annotation
|-- preprocessing/    # radiograph enhancement and ROI preparation
|-- segmentation/     # reusable mask-construction algorithms
|-- features/         # midline extraction and batch feature export
|-- evaluation/       # segmentation evaluation
|-- training/         # model-ready dataset validation
|-- models/           # working-length regression experiments
|-- prediction/       # inference entry points
`-- visualization/    # dataset plots
```

The tooth-isolation sequence is:

```text
tooth ROI
  -> metal removal and thresholding
  -> distance-transform core extraction
  -> central target-core selection
  -> competitive separation from neighbouring cores
  -> target-core reconstruction
  -> boundary refinement
  -> background removal
  -> tooth-midline extraction
```

Competitive reconstruction lives in
`src/working_length/segmentation/hybrid_core.py`. It separates structures that
thresholding merged by assigning candidate pixels to the closest tooth core.
It does not crop the input image or restrict it to a fixed-width band.

Run batch extraction with:

```bash
python -m src.working_length.features.extract_features --save
```

Use `--debug` to inspect the selected-core territory and competitive
reconstruction stages. The algorithm extracts the centreline of the segmented
tooth silhouette, not the anatomical root-canal centreline.

## Canal Detection

Canal detection is implemented independently under `src/canalwidth/`. It reads
the isolated tooth images produced by the working-length pipeline:

```text
data/working_length/segmentation/isolated_teeth/
```

Run it with:

```bash
python -m src.canalwidth.detect_canals --debug
```

The stages are CLAHE enhancement, black-hat filtering, adaptive thresholding,
small-kernel morphology, and central vertical-component selection. Outputs are
written under `data/canalwidth/segmentation/`:

```text
segmentation/
|-- masks/                 # binary canal masks
|-- overlays/              # red canal masks over isolated teeth
|-- debug/                 # intermediate stages and component reports
`-- canal_detection.csv    # detection status and mask area
```

Extract branch-free canal centrelines with:

```bash
python -m src.canalwidth.extract_midlines --debug
```

This stage supplements the usually dark canal mask with narrow central
radiopaque filling material, applies connectivity-preserving skeletonization,
keeps the longest graph path to remove branches, and fits a parametric spline.
Results are written under `data/canalwidth/midlines/`, including binary
midlines, overlays, measurements, ordered point coordinates, and debug stages.

Prepare the matched clinical dataset and fit the unchanged ordinary linear
regression model with leave-one-out validation:

```bash
python -m src.canalwidth.models.train_wl_model
```

Predict working lengths for valid canal midlines:

```bash
python -m src.canalwidth.predict_wl
```

Measure apical canal widths perpendicular to the centreline at 95%, 96%, 97%,
98%, and 99% of its arc length:

```bash
python -m src.canalwidth.measure_widths
```

The median of at least three valid cross-sections is retained. Outputs under
`data/canalwidth/widths/` include the median feature, every left-wall/centre/
right-wall measurement, and overlays showing the local perpendiculars.

Create radiographic apical-diameter estimates:

```bash
python -m src.canalwidth.estimate_apical_diameter
```

Estimates remain in pixels unless a valid calibration is provided with
`--mm-per-pixel`. Train the continuous first-binding-file regression using
leakage-safe out-of-fold working-length predictions:

```bash
python -m src.canalwidth.models.train_binding_file_model
```

Generate continuous predictions and nearest-ISO recommendations:

```bash
python -m src.canalwidth.predict_binding_file
```

The ISO mapping uses sizes 06, 08, 10, then 15–60 in increments of five and
the standard larger sizes through 140. Model metrics must be reviewed before
recommendations are used; the current dataset is exploratory and too small for
clinical deployment.

Map continuous file-size predictions to an explicit master-GP preparation
protocol:

```bash
python -m src.canalwidth.recommend_gp --protocol nearest-06
```

Available protocols are `nearest-04`, `upsize-04`, `nearest-06`, and
`upsize-06`. For a continuous size of 22, `nearest-06` returns `20/.06`, while
`upsize-04` returns `25/.04`. The software does not infer the preparation
system; the operator must select the applicable protocol.

## Working-Length API

Install the API upload dependency and start the FastAPI application:

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --reload
```

The workflow is split into common case/selection APIs and diagnosis-module APIs:

1. `POST /api/v1/cases` with multipart field `image`.
2. Display the returned `selection_image_url`.
3. Save the doctor's two selected points with
   `POST /api/v1/cases/{case_id}/tooth-selection`.
4. Call `POST /api/v1/cases/{case_id}/modules/working-length`. The fracture
   and lesion module routes use the same saved selection and tooth ROI.

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Upload a radiograph:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cases \
  -F "image=@radiograph.png"
```

The response includes a `case_id` and a 256x256 `selection_image_url`. Display
that exact image in the client and collect:

- the target-tooth centre;
- a second point indicating the tooth's vertical direction.

Submit the selection:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/v1/cases/CASE_ID/tooth-selection \
  -H "Content-Type: application/json" \
  -d '{
    "selected_x": 128,
    "selected_y": 120,
    "direction_x": 128,
    "direction_y": 200
  }'
```

The common endpoint persists `tooth_selection.json` and
`selected_tooth_roi.png`. Run working-length analysis without resending the
coordinates:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/v1/cases/CASE_ID/modules/working-length \
  -H "Content-Type: application/json" \
  -d '{"gp_protocol": "nearest-06"}'
```

The synchronous response contains the tooth and canal lengths, predicted
working length, apical diameter, file-size prediction, GP recommendation, and
URLs for all diagnostic images. Pipeline failures return HTTP 422 with the
failed stage and reason. Uploaded cases and their artifacts are isolated under
`data/api/cases/{case_id}/`.

The separation is implemented in `api/routes/common.py` and
`api/services/common_service.py`; diagnosis services should import
`load_saved_tooth_roi` from the common service.
