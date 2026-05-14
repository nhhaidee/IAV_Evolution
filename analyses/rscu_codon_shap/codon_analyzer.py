import numpy as np
from collections import Counter, defaultdict
import re
import warnings


class CodonAnalyzer:
    def __init__(self):

        # ignore stop codons in genetic code
        self.genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y',
            'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        self.stop_codons = {'TAA', 'TAG', 'TGA'}
        self.valid_codons = set(self.genetic_code.keys())
        self.synonymous_codon_groups = defaultdict(list)
        self.aa_groups = defaultdict(list)

        for codon, amino_acid in self.genetic_code.items():
            self.aa_groups[amino_acid].append(codon)
            # if amino_acid != 'M' and amino_acid != 'W':
            #     self.synonymous_codon_groups[amino_acid].append(codon)

    @staticmethod
    def extract_coding_sequence(sequence, cds_coordinates):
        if isinstance(cds_coordinates, str):
            if 'join' in cds_coordinates:
                # Parse join(28..252,740..1004) format
                coords = re.findall(r'(\d+)\.\.(\d+)', cds_coordinates)
                cds_sequence = ''
                for start, end in coords:
                    start, end = int(start) - 1, int(end)
                    cds_sequence += sequence[start:end]
                return cds_sequence
            else:
                start, end = map(int, cds_coordinates.split('..'))
                return sequence[start - 1:end]

    # Get header for table from genetic code with sorting accoring to amino acids and sort alphabetically within each animo acid group
    def get_header(self):
        """Get header for table from genetic code with sorting according to amino acids and sort alphabetically within each amino acid group"""
        header = []
        for aa, codons in sorted(self.aa_groups.items()):
            header.extend(sorted(codons))
        return header

    def codon_to_aa(self, codon):
        """Convert a codon to its corresponding amino acid"""
        return self.genetic_code.get(codon, 'X')

    def translate_sequence(self, cds_sequence):
        """Translate a nucleotide sequence into an amino acid sequence"""
        amino_acids = ''
        for i in range(0, len(cds_sequence), 3):
            codon = cds_sequence[i:i + 3]
            if codon not in self.stop_codons:
                amino_acid = self.codon_to_aa(codon)
                amino_acids += amino_acid
        return amino_acids

    def count_codons(self, sequence_id, sequence, cds_coordinates, codon_start=1):
        """Count codons in the given coding sequence"""
        full_cds_sequence = ""

        # Step 1: Correctly parse ALL start and end coordinates from the string.
        coords = re.findall(r'[<>]?(\d+)\.\.[<>]?(\d+)', cds_coordinates)
        if not coords:
            raise ValueError(f"Could not parse CDS coordinates from string: {cds_coordinates}")

        # Step 2: Build the complete CDS by concatenating all exon parts.
        for start, end in coords:
            start_index, end_index = int(start) - 1, int(end)
            full_cds_sequence += sequence[start_index:end_index]

        # Step 3: Apply the codon_start offset to the ASSEMBLED sequence.
        offset = int(codon_start) - 1
        trimmed_sequence = full_cds_sequence[offset:]

        # Step 4: Process the final sequence
        remainder = len(trimmed_sequence) % 3
        #if remainder != 0:
#            print(
#                f"{sequence_id} Codon counts: CDS length ({len(trimmed_sequence)}) is not a multiple of 3. "
#                f"Truncating last {remainder} base(s)."
#            )
#            warnings.warn(
#                f"{sequence_id} Codon counts: CDS length ({len(trimmed_sequence)}) is not a multiple of 3. "
#                f"Truncating last {remainder} base(s).",
#                UserWarning
#            )
        seq_length = len(trimmed_sequence) - remainder

        valid_codons = [
            trimmed_sequence[i:i + 3]
            for i in range(0, seq_length, 3)
            if trimmed_sequence[i:i + 3] in self.valid_codons
        ]

        #amino_acids_seq = self.translate_sequence(trimmed_sequence[:seq_length])

        if not valid_codons:
            raise ValueError("No valid codons found in the sequence")
        return Counter(valid_codons)
        #return Counter(valid_codons), trimmed_sequence[:seq_length], amino_acids_seq

    def calculate_rscu(self, codon_counts):
        # Relative Synonymous Codon Usage (RSCU)
        rscu_values = {}
        for aa, codons_in_group in self.aa_groups.items():
            # Count total occurrences of this amino acid
            total_observed_aa = sum(codon_counts.get(codon, 0) for codon in codons_in_group)
            num_synonymous_codons = len(codons_in_group)
            expected_freq = total_observed_aa / num_synonymous_codons  # codons of W and M have either value of 0 or 1
            for codon in codons_in_group:
                observed_freq = codon_counts.get(codon, 0)
                rscu_values[codon] = observed_freq / expected_freq if expected_freq > 0 else 0
        for codon in self.valid_codons:
            if codon not in rscu_values:
                rscu_values[codon] = 0.0#np.nan  # Handle missing codons
        sort_rscu_values = dict(
            sorted(rscu_values.items(), key=lambda item: (self.genetic_code.get(item[0], 'Z'), item[0]))
        )
        return sort_rscu_values

    def calculate_codon_shap_scores(self, sequence_id, sequence, cds_coordinates, shap_values, codon_start='', codon_investigate=None):


        # Step 1: Parse all coordinate pairs, ignoring optional '<' and '>' symbols.
        coords = re.findall(r'[<>]?(\d+)\.\.[<>]?(\d+)', cds_coordinates)
        if not coords:
            raise ValueError(f"Could not parse CDS coordinates from string: {cds_coordinates}")

        # Step 2: Build the complete CDS and corresponding SHAP array by concatenating all parts.
        cds_parts  = [sequence[int(start)-1:int(end)] for start, end in coords]
        shap_parts = [shap_values[:, int(start)-1:int(end)] for start, end in coords]
        
        full_cds_sequence = "".join(cds_parts)
        full_shap_values = np.concatenate(shap_parts, axis=1)

        # Step 3: Apply the codon_start offset to both the sequence and the SHAP array.
        if codon_start != '':
            offset = int(codon_start) - 1
        else:
            offset = 0
        trimmed_sequence = full_cds_sequence[offset:]
        trimmed_shaps = full_shap_values[:, offset:]

        # Step 4: Warn and truncate if the final length isn't a multiple of 3.
        remainder = len(trimmed_sequence) % 3
        #if remainder != 0:
#            print(
#                f"{sequence_id} Codon Shap score: CDS length ({len(trimmed_sequence)}) is not a multiple of 3. "
#                f"Truncating last {remainder} base(s)."
#            )
#            warnings.warn(
#                f"{sequence_id} Codon Shap score: CDS length ({len(trimmed_sequence)}) is not a multiple of 3. "
#                f"Truncating last {remainder} base(s).",
#                UserWarning
#            )
        seq_length = len(trimmed_sequence) - remainder

        # Step 5: Iterate through the final sequence and SHAP array to calculate scores.
        codon_shap_scores = {}
        codon_investigate_shap_scores = {}
        for i in range(0, seq_length, 3):
            codon = trimmed_sequence[i:i + 3]
            if codon in self.valid_codons:
                # Sum all SHAP scores for the 3 bases of the codon across all classes.
                codon_shap_sum = np.sum(trimmed_shaps[:, i:i + 3])
                codon_shap_scores[codon] = codon_shap_scores.get(codon, 0.0) + codon_shap_sum
                if codon == codon_investigate:
                    if codon not in codon_investigate_shap_scores:
                        codon_investigate_shap_scores[codon] = [np.sum(trimmed_shaps[:, i:i + 3], axis=0)]
                    else:
                        codon_investigate_shap_scores[codon].append(np.sum(trimmed_shaps[:, i:i + 3], axis=0))
        
        final_codon_investigate_shap_scores = {}
        if codon_investigate is not None and codon_investigate in codon_investigate_shap_scores.keys():
            if len(codon_investigate_shap_scores[codon_investigate]) > 1:
                tmp_shap_arr = np.array(codon_investigate_shap_scores[codon_investigate])
                sum_along_codon = tmp_shap_arr.sum(axis=0)
            else:
                sum_along_codon = codon_investigate_shap_scores[codon_investigate][0]
            for codon_pos, codon_nt in enumerate(codon_investigate):
                final_codon_investigate_shap_scores[f'{codon_nt}_{codon_pos}'] = sum_along_codon[codon_pos]    
                   
        # Step 6: Handle missing codons and sort the results.
        for codon in self.valid_codons:
            if codon not in codon_shap_scores:
                codon_shap_scores[codon] = 0.0#np.nan

        sorted_codon_shap_scores = dict(
            sorted(codon_shap_scores.items(), key=lambda item: (self.genetic_code.get(item[0], 'Z'), item[0]))
        )
        return sorted_codon_shap_scores, final_codon_investigate_shap_scores

