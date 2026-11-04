import json

import score

MK_SPLIT = """SM partition (explicit 111/21/0)
steady state: 40.00 ms/step (median 39.90) -> 102400 tok/s;  4,000 steps = 160.0 s
decode: 120000 tokens generated during 160.00 s of training -> 750 tok/s
loss: step0 11.7000  s1 11.0000  s3999 2.1500
"""


def write(d, name, obj):
    (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_parse_mk_log():
    r = score.parse_mk_log(MK_SPLIT)
    assert r["train_tok_s"] == 102400 and r["serve_tok_s"] == 750 and r["loss_final"] == 2.15 and r["errors"] == 0
    assert score.parse_mk_log("crash")["errors"] == 1


def test_two_jobs_cap_and_gates(tmp_path):
    write(tmp_path, "mk_split.log", MK_SPLIT)
    write(tmp_path, "mk_train.log", MK_SPLIT.replace("102400", "120000"))
    write(tmp_path, "pt_graph.json", {"tok_s": 110000.0, "seconds": 150.0, "loss_final": 2.145})
    write(tmp_path, "serve_full.json", {"tok_s": [1000.0, 1000.0]})
    write(tmp_path, "pt_mps_pair.json", {"tok_s": 90000.0, "loss_final": 2.146})
    write(tmp_path, "serve_mps_pair.json", {"tok_s": [500.0]})
    s = score.main([str(tmp_path)])
    row = s["fused"]["mk_split"]
    assert abs(row["useful_work"] - (102400 / 120000 + 0.75)) < 1e-9
    assert row["vs_serial"] < 2.0 and s["verdict"] is False
    relaxed = score.summarize(score.collect(tmp_path), dict(score.GATES, vs_serial_min=1.5, vs_mps_min=1.0))
    assert relaxed["verdict"] is True
    assert (tmp_path / "summary.json").exists()


def test_missing_baselines(tmp_path):
    write(tmp_path, "mk_split.log", MK_SPLIT)
    s = score.summarize(score.collect(tmp_path))
    assert s["errors"] and s["verdict"] is False


def test_dynamic_arm_credit(tmp_path):
    base = MK_SPLIT.replace("102400", "120000")
    paced = MK_SPLIT + ("serve: demand 200.0 tok/s (open-loop), 31000 tokens in the training window 155.00 s -> 200.0 tok/s, "
                        "latency p50 0.900 ms p99 2.100 ms, slo<=50ms 1.0000, lent train tasks 0\n")
    dyn = paced.replace("102400", "115000").replace("lent train tasks 0", "lent train tasks 52000")
    write(tmp_path, "mk_train.log", base)
    write(tmp_path, "mk_paced_200.log", paced)
    write(tmp_path, "mk_dyn_200.log", dyn)
    write(tmp_path, "pt_graph.json", {"tok_s": 110000.0, "loss_final": 2.145})
    write(tmp_path, "serve_full.json", {"tok_s": [1000.0]})
    write(tmp_path, "pt_mps_pair.json", {"tok_s": 90000.0})
    write(tmp_path, "serve_mps_pair.json", {"tok_s": [500.0]})
    s = score.summarize(score.collect(tmp_path))
    dyn_row = s["fused"]["mk_dyn_200"]
    assert abs(dyn_row["serve_frac"] - 0.2) < 1e-9 and dyn_row["lent_tasks"] == 52000
    assert dyn_row["vs_static_same_demand"] > 1.0
    assert abs(dyn_row["mps_useful_work_same_demand"] - (90000 / 120000 + 0.2)) < 1e-9
    assert s["verdict"] is False
