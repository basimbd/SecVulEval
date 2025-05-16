import os
import json
import random
import argparse
import subprocess
from datasets import load_dataset
from utils import get_json_file_as_dict, save_dict_as_json

def get_repo_path(data: dict):
    proj = data['project']
    if proj == "Chrome":
        return "symbol_backend_projects/chromium"
    if proj == "OpenJK":
        project_map = get_json_file_as_dict("symbol_backend_projects/project_links.json")
        url = project_map.get(proj, "")
        return "symbol_backend_projects/iortcw" if "iortcw" in url else "symbol_backend_projects/ioq3"
    return f"symbol_backend_projects/{proj}"

def get_top_25_cwe(full_dataset):
    cwe_vuls_list = {}
    used_commits = set()
    for vul_data in full_dataset:
        for cwe in vul_data["cwe_list"]:
            if cwe in ["NVD-CWE-noinfo", "NVD-CWE-Other"]:
                continue
            commit_id = vul_data["commit_id"]
            filepath = vul_data["filepath"]
            project = vul_data["project"]
            project_repo_path = get_repo_path(vul_data)
            is_vulnerable = vul_data["is_vulnerable"]
            if f"{commit_id}:{is_vulnerable}" in used_commits:
                continue        # to not keep collecting from similar changes
            cwe_vuls_list[cwe] = cwe_vuls_list.get(cwe, [])
            cwe_vuls_list[cwe].append(dict(
                commit_id=commit_id,
                filepath=filepath,
                project=project,
                project_repo_path=project_repo_path,
                is_vulnerable=is_vulnerable,
                func_name=vul_data["func_name"],
                func_body=vul_data["func_body"],
                line_statements=json.loads(vul_data["changed_lines"]),
                statements=json.loads(vul_data["changed_statements"]),
                cve=vul_data["cve_list"],
            ))
            used_commits.add(f"{commit_id}:{is_vulnerable}")

    # top = 1
    # for cwe_id, func_list in sorted(cwe_vuls_list.items(), key=lambda item: len(item[1]), reverse=True):
    #     if top > 25:
    #         break
    #     # print(f"{top} - {cwe_id}: {len(func_list)}")
    #     top += 1

    # print("Total:", len(cwe_vuls_list))
    return cwe_vuls_list

def clone_repository(vul_list: list):
    project_map = get_json_file_as_dict("symbol_backend_projects/project_links.json")
    for vul_data in vul_list:
        project = vul_data["project"]
        project_repo_path = get_repo_path(vul_data)
        if not os.path.exists(project_repo_path):
            print(f"Cloning {project}...")
            cwd = os.getcwd()
            try:
                os.chdir("symbol_backend_projects")
                subprocess.run(["git", "clone", project_map[project]], check=True)
            except Exception as e:
                print(f"Error cloning {project}: {e}")
            os.chdir(cwd)
        else:
            print(f"{project} already cloned.")


def save_random_subset(size: int = 300):
    full_dataset = load_dataset("arag0rn/SecVulEval", split="train")
    cwe_vuls_list = get_top_25_cwe(full_dataset)
    # select proportional samples for each CWE type
    # select size samples.
    # (size * func_of_that_cwe) / total_samples
    top = 1
    top_25_cwe_types = {}
    for cwe_id, func_list in sorted(cwe_vuls_list.items(), key=lambda item: len(item[1]), reverse=True):
        if top > 25:
            break
        top += 1
        top_25_cwe_types[cwe_id] = func_list

    sample_cnt_per_cwe = {k: 0 for k in top_25_cwe_types.keys()}
    selected_samples = {k: [] for k in top_25_cwe_types.keys()}
    total_samples = sum(len(v) for v in top_25_cwe_types.values())
    for cwe_id, func_list in top_25_cwe_types.items():
        sample_cnt_per_cwe[cwe_id] = round(len(func_list) * size / total_samples)

    for cwe_id, func_list in top_25_cwe_types.items():
        selected_samples[cwe_id] = random.sample(func_list, sample_cnt_per_cwe[cwe_id])
        clone_repository(selected_samples[cwe_id])
    # pprint.pprint(selected_samples)
    print(sum([len(v) for v in selected_samples.values()]))
    save_dict_as_json("random_subset_.json", selected_samples, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get random subset of vulnerabilities")
    parser.add_argument("--size", type=int, default=300, help="Size of the random subset")
    args = parser.parse_args()

    save_random_subset(args.size)
