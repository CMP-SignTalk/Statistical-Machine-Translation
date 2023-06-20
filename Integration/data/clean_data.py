import json
import re

def gloss_to_min_gloss(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            processed_line = ' '.join([word.split('-')[1] if '-' in word else word for word in line.split()])
            processed_line = processed_line.lower()
            outfile.write(processed_line + '\n')

def merge_en_asl_files(en_file, asl_file, output_file):
    with open(en_file, 'r', encoding='utf-8') as en_file:
        en_lines = en_file.readlines()

    with open(asl_file, 'r', encoding='utf-8') as asl_file:
        asl_lines = asl_file.readlines()

    merged_data = []
    count = 1

    for en_line, asl_line in zip(en_lines, asl_lines):
        en_entry = en_line.strip().lower()
        asl_entry = ' '.join([word.split('-')[1] if '-' in word else word for word in asl_line.strip().lower().split()])
        # Remove special characters with regex
        en_entry = re.sub(r'[^a-zA-Z0-9\s]', '', en_entry)
        asl_entry = re.sub(r'[^a-zA-Z0-9\s]', '', asl_entry)
        asl_entry = asl_entry.strip()
        merged_entry = {
            'id' :  count,
            'en': en_entry,
            'asl': asl_entry
        }
        merged_data.append(merged_entry)
        count += 1

    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, ensure_ascii=False, indent=2)

merge_en_asl_files('aslg.small.en', 'aslg.small.gloss.asl', 'small.json')
merge_en_asl_files('aslg.train.en', 'aslg.train.gloss.asl', 'train.json')
merge_en_asl_files('aslg.dev.en', 'aslg.dev.gloss.asl', 'dev.json')
merge_en_asl_files('aslg.test.en', 'aslg.test.gloss.asl', 'test.json')
