import re
from collections import defaultdict


def parse_vadr_cds_tbl(file_path):
    sequences = {}
    current_seq_id = None
    current_feature = None

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                # Check for a new sequence feature line (e.g., >Feature EPI2041961)
                if line.startswith('>Feature'):
                    parts = line.strip().split()
                    if len(parts) > 1:
                        current_seq_id = parts[1]
                        sequences[current_seq_id] = []
                        current_feature = None
                # Check for a qualifier line (indented with tabs)
                elif line.startswith('\t'):
                    if current_feature:
                        qualifier_parts = line.strip().split('\t')
                        if len(qualifier_parts) == 2:
                            key, value = qualifier_parts
                            current_feature['qualifiers'][key] = value
                # Otherwise, it's a feature or a continuation of coordinates
                else:
                    feature_parts = line.split('\t')
                    # A new feature is being defined (e.g., "26  51  CDS")
                    if len(feature_parts) == 3 and current_seq_id:
                        start, end, key = feature_parts
                        feature = {
                            'key': key,
                            'coordinates': [(start, end)],  # Store coordinates as a list of tuples
                            'qualifiers': {}
                        }
                        sequences[current_seq_id].append(feature)
                        current_feature = feature
                    # A continuation of the previous feature (e.g., "740 1007")
                    elif len(feature_parts) == 2 and current_feature:
                        start, end = feature_parts
                        # Append new coordinate pair to the current feature
                        current_feature['coordinates'].append((start, end))

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

    return sequences


def format_cds_coordinates(parsed_data):

    formatted_output = defaultdict(dict)
    if not parsed_data:
        return {}

    for seq_id, features in parsed_data.items():
        cds_info_by_product = defaultdict(lambda: {'coords': [], 'codon_start': None})

        for feature in features:
            if feature['key'] == 'CDS':
                product = feature['qualifiers'].get('product', 'unknown_product')
                cds_info_by_product[product]['coords'].extend(feature['coordinates'])

                if cds_info_by_product[product]['codon_start'] is None:
                    codon_start = feature['qualifiers'].get('codon_start')
                    if codon_start:
                        cds_info_by_product[product]['codon_start'] = codon_start

        for product, info in cds_info_by_product.items():
            coords_list = info['coords']
            codon_start_val = info['codon_start']

            coords_list.sort(key=lambda x: int(x[0].strip('<>')))

            coord_strings = [f"{start}..{end}" for start, end in coords_list]
            formatted_coord = f"join({','.join(coord_strings)})" if len(coord_strings) > 1 else coord_strings[0]

            # Change #2: Build a details dictionary without the 'product' key
            product_details = {
                'coordinate': formatted_coord
            }
            if codon_start_val:
                product_details['codon_start'] = codon_start_val

            # Change #3: Assign the details dict to its product key
            formatted_output[seq_id][product] = product_details

    return dict(formatted_output)

