import json

with open('DescartesMMRec_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Fix cell 2 (git clone / pull)
            if 'if not os.path.exists(WORK_DIR):' in line:
                source[i] = "if os.path.exists(WORK_DIR):\n"
                source.insert(i+1, "    print('Folder đã tồn tại, đang pull bản mới nhất...')\n")
                source.insert(i+2, "    !cd {WORK_DIR} && git pull\n")
                source.insert(i+3, "else:\n")
                break

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Fix cell 5 (add model_type)
            if 'args_str += f"--dataset_path {DATASET_DRIVE_PATH}' in line:
                source[i] = line.replace('}', '} --model_type descartes_mmrec')
                break

with open('DescartesMMRec_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
