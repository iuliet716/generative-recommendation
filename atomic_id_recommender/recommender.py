from datetime import datetime
import json
import os

from recbole.quick_start import run_recbole

def execute(model, dataset_name, data_path, config_path, result_path):
    result = run_recbole(
        model=model,
        dataset=dataset_name,
        config_file_list=[config_path],
        config_dict={"data_path": data_path},
    )

    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{dataset_name}_{model}_result.json")

    record = {
        "model": model,
        "dataset": dataset_name,
        "id_type": "atomic",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_valid_result": dict(result["best_valid_result"]),
        "test_result": dict(result["test_result"]),
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[result] {model} on {dataset_name}: {record['test_result']}")
    print(f"[result] saved -> {output_path}")
    
    return result

