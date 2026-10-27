"""Export a reproducible experiment and a standalone interactive result page."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from battery_model import Config, METHODS, simulate


def export(folder, include_dfn=False):
    folder.mkdir(parents=True, exist_ok=True)
    c = Config()
    results = {m: simulate(c, m) for m in METHODS}
    summary = pd.DataFrame([r.summary() for r in results.values()])
    summary.to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([r.data.assign(method=m) for m, r in results.items()]).to_csv(
        folder / "timeseries.csv", index=False, encoding="utf-8-sig"
    )
    for m, r in results.items():
        np.savez_compressed(
            folder / f"concentration_{m}.npz",
            time_s=r.data.time_s,
            x_m=(np.arange(c.nodes) + 0.5) * c.length_m / c.nodes,
            concentration_mol_m3=r.profiles,
        )
    source = Path(__file__).resolve().parent
    metadata = {
        "config": asdict(c),
        "model": "uncalibrated reduced-order diffusion/thermal/control",
        "source_sha256": {
            f: hashlib.sha256((source / f).read_bytes()).hexdigest()
            for f in ["battery_model.py", "app.py", "dfn_validation.py"]
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "streamlit", "plotly"]
        },
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sensitivity = []
    for alpha in [0.0, 1.0, 2.0]:
        sensitivity.append(
            {
                "alpha": alpha,
                **simulate(replace(c, dispersion_gain=alpha), "combined").summary(),
            }
        )
    pd.DataFrame(sensitivity).to_csv(
        folder / "field_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    convergence = []
    for label, conf in [
        ("base", c),
        ("dt/2", replace(c, dt=c.dt / 2)),
        ("nodes*2", replace(c, nodes=c.nodes * 2)),
    ]:
        r = simulate(conf, "combined")
        convergence.append(
            {
                "case": label,
                **r.summary(),
                "charge_residual_ah": r.data.charge_residual_ah.abs().max(),
                "salt_residual_mol_m2": r.data.salt_residual_mol_m2.abs().max(),
            }
        )
    pd.DataFrame(convergence).to_csv(
        folder / "convergence.csv", index=False, encoding="utf-8-sig"
    )
    fields = [
        ("soc", "SOC (%)", 100),
        ("plated_ah", "누적 석출 (mAh)", 1000),
        ("temperature_c", "온도 (°C)", 1),
        ("voltage_v", "단자 전압 (V)", 1),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[x[1] for x in fields])
    palette = ["#ef6c63", "#4ba3d3", "#bd93e8", "#e6aa46", "#23b59f"]
    for i, (field, label, scale) in enumerate(fields):
        row, col = i // 2 + 1, i % 2 + 1
        for (m, r), color in zip(results.items(), palette):
            fig.add_trace(
                go.Scatter(
                    x=r.data.time_s / 60,
                    y=r.data[field] * scale,
                    name=METHODS[m],
                    legendgroup=m,
                    showlegend=i == 0,
                    line=dict(color=color),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="경과 시간 (분)", row=row, col=col)
        fig.update_yaxes(title_text=label, row=row, col=col)
    fig.update_layout(
        height=850,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=65),
    )
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>배터리 충전 비교 결과</title><style>body{font-family:system-ui,sans-serif;margin:35px;color:#172435}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ddd}p{line-height:1.6}.note{padding:16px;background:#eef4fa}</style>'
    page += '<h1>자기장 · 적응형 펄스 충전 비교</h1><p class="note">연구용 가정 모델의 예시 결과입니다. 실제 덴드라이트 형상·수명·화재 확률을 예측하지 않습니다. 모든 방식은 같은 초기 10% → 목표 80% SOC, 최대 3C, 전압 4.2 V와 온도 45°C 제한을 사용합니다. 모델 식과 가정은 코드의 MODEL_NOTES.md에 있습니다.</p>'
    page += summary.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    page += fig.to_html(full_html=False, include_plotlyjs=True)
    page += "<p>석출 감소·충전 시간·전자석 에너지의 차이를 함께 해석하세요. 듀티비는 에너지 효율이 아닙니다. 원시 값과 매개변수·소스 해시는 함께 저장한 CSV와 metadata.json에 있습니다.</p></html>"
    (folder / "comparison.html").write_text(page, encoding="utf-8")
    if include_dfn:
        from dfn_validation import run_comparison

        frames, table = run_comparison(c)
        table.to_csv(folder / "dfn_summary.csv", index=False, encoding="utf-8-sig")
        pd.concat([frame.assign(method=name) for name, frame in frames.items()]).to_csv(
            folder / "dfn_timeseries.csv", index=False, encoding="utf-8-sig"
        )
        print(table.to_string(index=False))
    print(summary.to_string(index=False))
    print("Exported:", folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dfn", action="store_true")
    args = parser.parse_args()
    export(args.output, args.dfn)
