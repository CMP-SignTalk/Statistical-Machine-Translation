with open('_aslg.asl', 'r') as infile, open('_aslg_processed.asl', 'w') as outfile:
    for line in infile:
        processed_line = ' '.join([word.split('-')[1] if '-' in word else word for word in line.split()])
        processed_line = processed_line.lower()
        outfile.write(processed_line + '\n')