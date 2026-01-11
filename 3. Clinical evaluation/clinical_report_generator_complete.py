# clinical_report_generator_fixed.py
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import json
import traceback

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClinicalReportGenerator:
    """Clinical Report Generator with Statistics and Visualization"""
    
    def __init__(self, output_base_dir: str = None):
        """Initialize generator with output directory"""
        if output_base_dir is None:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base_dir = f"D:/MAMA-MIA/report_generation_{timestamp}"
        
        self.output_base_dir = output_base_dir
        self._init_directories()
        self._init_terminology_mappings()
        self._init_statistics()
        
        print(f"Output directory: {self.output_base_dir}")
    
    def _init_directories(self):
        """Initialize output directory structure"""
        self.dirs = {
            'reports': os.path.join(self.output_base_dir, 'structured_reports'),
            'bert_texts': os.path.join(self.output_base_dir, 'bert_texts'),
            'bert_individual': os.path.join(self.output_base_dir, 'bert_individual_files'),
            'statistics': os.path.join(self.output_base_dir, 'statistics'),
            'visualizations': os.path.join(self.output_base_dir, 'visualizations'),
            'metadata': os.path.join(self.output_base_dir, 'metadata'),
            'latex_tables': os.path.join(self.output_base_dir, 'latex_tables')
        }
        
        # Create all directories
        for dir_name, dir_path in self.dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"  Created: {dir_path}")
    
    def _init_terminology_mappings(self):
        """Initialize clinical terminology mappings"""
        # Status mapping
        self.status_mapping = {
            'positive': 'positive', 'pos': 'positive', '1': 'positive', 'yes': 'positive', 'y': 'positive',
            'negative': 'negative', 'neg': 'negative', '0': 'negative', 'no': 'negative', 'n': 'negative',
            'equivocal': 'equivocal', 'borderline': 'borderline'
        }
        
        # Subtype mapping
        self.subtype_mapping = {
            'luminal_a': 'Luminal A', 'luminal a': 'Luminal A',
            'luminal_b': 'Luminal B', 'luminal b': 'Luminal B',
            'her2_enriched': 'HER2-enriched', 'her2 enriched': 'HER2-enriched',
            'triple_negative': 'triple-negative', 'triple negative': 'triple-negative',
            'tnbc': 'triple-negative',
            'her2_pure': 'HER2-pure'
        }
        
        # Ethnicity mapping
        self.ethnicity_mapping = {
            'asian': 'Asian', 'caucasian': 'Caucasian', 'white': 'Caucasian',
            'african': 'African', 'black': 'African', 'african_american': 'African American',
            'hispanic': 'Hispanic', 'latino': 'Hispanic'
        }
        
        # Menopause mapping
        self.menopause_mapping = {
            'premenopausal': 'premenopausal', 'pre': 'premenopausal',
            'postmenopausal': 'postmenopausal', 'post': 'postmenopausal',
            'perimenopausal': 'perimenopausal', 'peri': 'perimenopausal'
        }
    
    def _init_statistics(self):
        """Initialize statistics collection"""
        self.stats = {
            'total_patients': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'missing_data_counts': {},
            'text_lengths': [],
            'subtype_distribution': {},
            'pcr_distribution': {'achieved': 0, 'not_achieved': 0, 'unknown': 0},
            'therapy_distribution': {},
            'field_completeness': {}
        }
    
    def _safe_get(self, value, default=None, return_raw=False):
        """Safely get value, handling NaN and None"""
        if pd.isna(value) or value is None:
            return default
        
        # Return raw value if requested
        if return_raw:
            return value
        
        # Convert to string and clean
        try:
            if isinstance(value, (int, np.integer)):
                return str(value)
            elif isinstance(value, (float, np.floating)):
                if value.is_integer():
                    return str(int(value))
                return f"{value:.2f}"
            else:
                return str(value).strip()
        except:
            return str(value).strip() if value else default
    
    def _parse_demographics(self, row: pd.Series) -> Dict:
        """Parse demographic information"""
        demo = {}
        
        # Age
        age = self._safe_get(row.get('age'), return_raw=True)
        if age is not None:
            try:
                demo['age'] = int(float(age))
                demo['age_str'] = f"{demo['age']}-year-old"
            except:
                demo['age_str'] = str(age)
        
        # Ethnicity
        ethnicity = self._safe_get(row.get('ethnicity'))
        if ethnicity:
            eth_lower = ethnicity.lower().replace(' ', '_')
            demo['ethnicity'] = self.ethnicity_mapping.get(eth_lower, ethnicity)
        
        # Menopause
        menopause = self._safe_get(row.get('menopause'))
        if menopause:
            meno_lower = menopause.lower()
            demo['menopause'] = self.menopause_mapping.get(meno_lower, menopause)
        
        # Tumor subtype
        subtype = self._safe_get(row.get('tumor_subtype'))
        if subtype:
            subtype_lower = subtype.lower().replace(' ', '_')
            demo['subtype'] = self.subtype_mapping.get(subtype_lower, subtype)
            
            # Update statistics
            if demo['subtype'] in self.stats['subtype_distribution']:
                self.stats['subtype_distribution'][demo['subtype']] += 1
            else:
                self.stats['subtype_distribution'][demo['subtype']] = 1
        
        # BMI group
        bmi = self._safe_get(row.get('bmi_group'))
        if bmi:
            demo['bmi'] = bmi
        
        # Weight and height
        weight = self._safe_get(row.get('weight'), return_raw=True)
        if weight is not None:
            try:
                demo['weight'] = f"{float(weight):.1f} kg"
            except:
                demo['weight'] = str(weight)
        
        size = self._safe_get(row.get('patient_size'), return_raw=True)
        if size is not None:
            try:
                demo['height'] = f"{float(size):.1f} cm"
            except:
                demo['height'] = str(size)
        
        return demo
    
    def _parse_molecular_status(self, row: pd.Series) -> Dict:
        """Parse molecular receptor status"""
        status = {}
        markers = ['hr', 'er', 'pr', 'her2']
        
        for marker in markers:
            val = self._safe_get(row.get(marker))
            if val:
                val_lower = str(val).lower()
                parsed = self.status_mapping.get(val_lower, val_lower)
                status[marker.upper()] = parsed
        
        return status
    
    def _parse_therapies(self, row: pd.Series) -> List[str]:
        """Parse therapy information"""
        therapies = []
        
        # Check endocrine therapy
        endocrine = self._safe_get(row.get('endocrine_therapy'))
        if endocrine:
            endocrine_lower = str(endocrine).lower()
            if endocrine_lower in ['yes', 'true', '1', 'y', 'positive']:
                therapies.append('endocrine therapy')
                self._update_therapy_stat('endocrine')
        
        # Check anti-HER2 therapy
        anti_her2 = self._safe_get(row.get('anti_her2_neu_therapy'))
        if anti_her2:
            anti_her2_lower = str(anti_her2).lower()
            if anti_her2_lower in ['yes', 'true', '1', 'y', 'positive']:
                therapies.append('anti-HER2 therapy')
                self._update_therapy_stat('anti-her2')
        
        # All patients have pCR status, so assume neoadjuvant chemo
        therapies.append('neoadjuvant chemotherapy')
        self._update_therapy_stat('chemotherapy')
        
        return therapies
    
    def _update_therapy_stat(self, therapy_type: str):
        """Update therapy distribution statistics"""
        if therapy_type in self.stats['therapy_distribution']:
            self.stats['therapy_distribution'][therapy_type] += 1
        else:
            self.stats['therapy_distribution'][therapy_type] = 1
    
    def _parse_outcomes(self, row: pd.Series) -> Dict:
        """Parse outcome information"""
        outcomes = {}
        
        # pCR status
        pcr = self._safe_get(row.get('pcr'))
        if pcr is not None:
            pcr_str = str(pcr).lower().strip()
            if pcr_str in ['1', 'yes', 'true', 'positive', 'y']:
                outcomes['pCR'] = 'achieved'
                self.stats['pcr_distribution']['achieved'] += 1
            elif pcr_str in ['0', 'no', 'false', 'negative', 'n']:
                outcomes['pCR'] = 'not achieved'
                self.stats['pcr_distribution']['not_achieved'] += 1
            else:
                outcomes['pCR'] = 'unknown'
                self.stats['pcr_distribution']['unknown'] += 1
        
        # Mastectomy
        mastectomy = self._safe_get(row.get('mastectomy_post_nac'))
        if mastectomy:
            mastectomy_lower = str(mastectomy).lower()
            if mastectomy_lower in ['yes', 'true', '1', 'y']:
                outcomes['mastectomy'] = 'Yes'
            elif mastectomy_lower in ['no', 'false', '0', 'n']:
                outcomes['mastectomy'] = 'No'
        
        # Time-based outcomes
        time_fields = {
            'days_to_follow_up': 'follow_up',
            'days_to_recurrence': 'recurrence',
            'days_to_death': 'death',
            'days_to_metastasis': 'metastasis'
        }
        
        for field, key in time_fields.items():
            val = self._safe_get(row.get(field), return_raw=True)
            if val is not None:
                try:
                    days = int(float(val))
                    outcomes[key] = days
                    outcomes[f'{key}_str'] = f"{days} days"
                except:
                    outcomes[key] = str(val)
        
        return outcomes
    
    def _generate_structured_report(self, patient_id: str, demographics: Dict, 
                                   molecular_status: Dict, therapies: List, 
                                   outcomes: Dict) -> str:
        """Generate structured clinical report"""
        
        sections = []
        
        # Header
        sections.append("BREAST CANCER CLINICAL DIAGNOSTIC REPORT")
        sections.append("=" * 60)
        
        # Patient Information
        sections.append("\nI. PATIENT INFORMATION")
        sections.append(f"   Patient ID: {patient_id}")
        
        if 'age_str' in demographics:
            sections.append(f"   Age: {demographics['age_str']}")
        elif 'age' in demographics:
            sections.append(f"   Age: {demographics['age']} years")
        
        for field in ['ethnicity', 'menopause', 'bmi', 'weight', 'height']:
            if field in demographics:
                field_name = field.replace('_', ' ').title()
                sections.append(f"   {field_name}: {demographics[field]}")
        
        # Tumor Characteristics
        sections.append("\nII. TUMOR CHARACTERISTICS")
        
        if 'subtype' in demographics:
            sections.append(f"   Molecular Subtype: {demographics['subtype']}")
        
        if molecular_status:
            status_lines = [f"{marker}: {status}" for marker, status in molecular_status.items()]
            sections.append(f"   Receptor Status: {', '.join(status_lines)}")
        
        # Treatment
        sections.append("\nIII. TREATMENT")
        if therapies:
            sections.append(f"   Therapies Received: {', '.join(therapies)}")
        else:
            sections.append("   Therapy Information: Not available")
        
        # Outcomes
        sections.append("\nIV. TREATMENT RESPONSE AND OUTCOMES")
        
        if 'pCR' in outcomes:
            pcr_status = outcomes['pCR'].replace('_', ' ').title()
            sections.append(f"   Pathological Complete Response (pCR): {pcr_status}")
        
        if 'mastectomy' in outcomes:
            sections.append(f"   Mastectomy post-NAC: {outcomes['mastectomy']}")
        
        # Time outcomes
        for key, display in [('follow_up_str', 'Follow-up Duration'), 
                            ('recurrence_str', 'Time to Recurrence'),
                            ('metastasis_str', 'Time to Metastasis'),
                            ('death_str', 'Survival Time')]:
            if key in outcomes:
                sections.append(f"   {display}: {outcomes[key]}")
        
        # Footer
        sections.append("\n" + "=" * 60)
        sections.append(f"Report Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}")
        sections.append("Generated by Clinical Report Generator v2.0")
        
        return "\n".join(sections)
    
    def _generate_bert_text(self, patient_id: str, demographics: Dict, 
                           molecular_status: Dict, therapies: List, 
                           outcomes: Dict) -> str:
        """Generate BERT-friendly text"""
        
        sentences = []
        
        # Demographic sentence
        demo_parts = []
        if 'age_str' in demographics:
            demo_parts.append(demographics['age_str'])
        if 'ethnicity' in demographics:
            demo_parts.append(demographics['ethnicity'])
        if 'menopause' in demographics:
            demo_parts.append(demographics['menopause'])
        
        if demo_parts:
            sentences.append(f"Patient is a {' '.join(demo_parts)} female.")
        
        # Diagnosis sentence
        diagnosis_parts = []
        if 'subtype' in demographics:
            diagnosis_parts.append(f"{demographics['subtype']} breast cancer")
        
        if molecular_status:
            markers = [f"{m} {s}" for m, s in molecular_status.items()]
            diagnosis_parts.append(f"receptor status: {', '.join(markers)}")
        
        if diagnosis_parts:
            sentences.append(f"Diagnosed with {' and '.join(diagnosis_parts)}.")
        
        # Treatment sentence
        if therapies:
            if len(therapies) == 1:
                sentences.append(f"Received {therapies[0]}.")
            else:
                sentences.append(f"Treatment includes {', '.join(therapies[:-1])} and {therapies[-1]}.")
        
        # pCR sentence
        if 'pCR' in outcomes:
            verb = "achieved" if outcomes['pCR'] == 'achieved' else "did not achieve"
            sentences.append(f"Patient {verb} pathological complete response (pCR) after neoadjuvant chemotherapy.")
        
        # Events sentence
        events = []
        for event, key in [('recurrence', 'recurrence'), 
                          ('metastasis', 'metastasis'), 
                          ('death', 'death')]:
            if key in outcomes:
                days = outcomes.get(key)
                events.append(f"{event} at {days} days" if isinstance(days, (int, float)) else event)
        
        if events:
            sentences.append(f"Clinical events: {'; '.join(events)}.")
        elif any(key in outcomes for key in ['follow_up', 'recurrence', 'metastasis', 'death']):
            sentences.append("No major clinical events recorded.")
        
        bert_text = " ".join(sentences)
        
        # Fallback if no text generated
        if not bert_text.strip():
            bert_text = f"Clinical information for patient {patient_id}."
        
        # Track text length for statistics
        self.stats['text_lengths'].append(len(bert_text))
        
        return bert_text
    
    def generate_patient_report(self, row: pd.Series) -> Tuple[str, str, Dict]:
        """Generate both report types for a single patient"""
        
        patient_id = self._safe_get(row.get('patient_id'), f"patient_{row.name}")
        
        try:
            # Parse all components
            demographics = self._parse_demographics(row)
            molecular_status = self._parse_molecular_status(row)
            therapies = self._parse_therapies(row)
            outcomes = self._parse_outcomes(row)
            
            # Generate reports
            structured_report = self._generate_structured_report(
                patient_id, demographics, molecular_status, therapies, outcomes
            )
            
            bert_text = self._generate_bert_text(
                patient_id, demographics, molecular_status, therapies, outcomes
            )
            
            # Track success
            self.stats['successful_generations'] += 1
            
            return structured_report, bert_text, {
                'patient_id': patient_id,
                'demographics_count': len(demographics),
                'molecular_markers': len(molecular_status),
                'therapies_count': len(therapies),
                'outcomes_count': len(outcomes),
                'error': None
            }
            
        except Exception as e:
            # Error handling
            error_msg = f"Error generating report: {str(e)}"
            print(f"  Warning: {error_msg} for patient {patient_id}")
            
            self.stats['failed_generations'] += 1
            
            # Generate minimal reports
            structured_report = f"Error generating report for patient {patient_id}. {error_msg}"
            bert_text = f"Clinical data for patient {patient_id} could not be processed."
            
            return structured_report, bert_text, {
                'patient_id': patient_id,
                'error': str(e)
            }
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process entire DataFrame and generate reports"""
        
        print(f"\nProcessing {len(df)} patients...")
        
        clinical_data = []
        bert_data = []
        
        # Initialize field completeness tracking
        for col in df.columns:
            self.stats['field_completeness'][col] = df[col].notna().sum()
        
        # Process each patient
        for idx, row in df.iterrows():
            if idx % 200 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(df)} patients...")
            
            structured_report, bert_text, metadata = self.generate_patient_report(row)
            
            patient_id = metadata['patient_id']
            
            # Save individual BERT file
            bert_file_path = os.path.join(self.dirs['bert_individual'], f"{patient_id}.txt")
            try:
                with open(bert_file_path, 'w', encoding='utf-8') as f:
                    f.write(bert_text)
            except Exception as e:
                print(f"  Warning: Could not save BERT file for {patient_id}: {e}")
            
            # Collect data
            clinical_data.append({
                'patient_id': patient_id,
                'clinical_report': structured_report,
                'report_length': len(structured_report),
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'has_error': metadata.get('error') is not None
            })
            
            bert_data.append({
                'patient_id': patient_id,
                'bert_text': bert_text,
                'text_length': len(bert_text),
                'word_count': len(bert_text.split()),
                'bert_file_path': bert_file_path,
                'has_error': metadata.get('error') is not None
            })
        
        # Create DataFrames
        clinical_df = pd.DataFrame(clinical_data)
        bert_df = pd.DataFrame(bert_data)
        
        # Update total count
        self.stats['total_patients'] = len(df)
        
        return clinical_df, bert_df
    
    def save_outputs(self, clinical_df: pd.DataFrame, bert_df: pd.DataFrame):
        """Save all output files"""
        
        print("\nSaving output files...")
        
        # 1. Save structured reports
        clinical_path = os.path.join(self.dirs['reports'], 'clinical_structured_reports.csv')
        clinical_df.to_csv(clinical_path, index=False, encoding='utf-8')
        print(f"  ✓ Structured reports: {clinical_path}")
        
        # 2. Save BERT texts index
        bert_path = os.path.join(self.dirs['bert_texts'], 'bert_input_texts.csv')
        bert_df.to_csv(bert_path, index=False, encoding='utf-8')
        print(f"  ✓ BERT texts index: {bert_path}")
        
        # 3. Generate and save BERT corpus
        corpus_path = os.path.join(self.dirs['bert_texts'], 'bert_corpus.txt')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for text in bert_df[~bert_df['has_error']]['bert_text']:
                f.write(text + '\n\n')
        print(f"  ✓ BERT corpus: {corpus_path}")
        
        # 4. Save metadata
        self._save_metadata(clinical_df, bert_df)
        
        return clinical_path, bert_path, corpus_path
    
    def _save_metadata(self, clinical_df: pd.DataFrame, bert_df: pd.DataFrame):
        """Save generation metadata"""
        
        metadata = {
            'generation_info': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_patients': self.stats['total_patients'],
                'successful': self.stats['successful_generations'],
                'failed': self.stats['failed_generations'],
                'success_rate': self.stats['successful_generations'] / self.stats['total_patients'] * 100
            },
            'text_statistics': {
                'avg_clinical_report_length': float(clinical_df['report_length'].mean()),
                'avg_bert_text_length': float(bert_df['text_length'].mean()),
                'min_bert_length': float(bert_df['text_length'].min()),
                'max_bert_length': float(bert_df['text_length'].max()),
                'total_words': int(bert_df['word_count'].sum())
            },
            'clinical_statistics': {
                'pcr_distribution': self.stats['pcr_distribution'],
                'subtype_distribution': self.stats['subtype_distribution'],
                'therapy_distribution': self.stats['therapy_distribution']
            },
            'field_completeness': {
                col: {
                    'non_null': int(count),
                    'completeness': float(count / self.stats['total_patients'] * 100)
                }
                for col, count in self.stats['field_completeness'].items()
            },
            'output_files': {
                'structured_reports': os.path.join(self.dirs['reports'], 'clinical_structured_reports.csv'),
                'bert_texts_index': os.path.join(self.dirs['bert_texts'], 'bert_input_texts.csv'),
                'bert_corpus': os.path.join(self.dirs['bert_texts'], 'bert_corpus.txt'),
                'bert_individual_files': self.dirs['bert_individual'],
                'statistics': self.dirs['statistics'],
                'visualizations': self.dirs['visualizations'],
                'latex_tables': self.dirs['latex_tables']
            }
        }
        
        # Save as JSON
        metadata_path = os.path.join(self.dirs['metadata'], 'generation_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        metadata_df = pd.DataFrame([{
            'total_patients': self.stats['total_patients'],
            'successful_generations': self.stats['successful_generations'],
            'failed_generations': self.stats['failed_generations'],
            'success_rate': f"{metadata['generation_info']['success_rate']:.1f}%",
            'avg_bert_length': f"{metadata['text_statistics']['avg_bert_text_length']:.1f}",
            'total_words': metadata['text_statistics']['total_words']
        }])
        metadata_df.to_csv(os.path.join(self.dirs['metadata'], 'summary_stats.csv'), index=False)
        
        print(f"  ✓ Metadata saved: {metadata_path}")
    
    def generate_statistics_and_plots(self, clinical_df: pd.DataFrame, bert_df: pd.DataFrame):
        """Generate statistics and visualization plots for paper"""
        
        print("\nGenerating statistics and visualizations...")
        
        # 1. Basic statistics
        basic_stats = {
            'Generation Summary': [
                ('Total Patients', self.stats['total_patients']),
                ('Successful Reports', self.stats['successful_generations']),
                ('Failed Reports', self.stats['failed_generations']),
                ('Success Rate', f"{self.stats['successful_generations']/self.stats['total_patients']*100:.1f}%")
            ],
            'Text Statistics': [
                ('Avg Clinical Report Length', f"{clinical_df['report_length'].mean():.0f} chars"),
                ('Avg BERT Text Length', f"{bert_df['text_length'].mean():.0f} chars"),
                ('Avg Word Count', f"{bert_df['word_count'].mean():.0f} words"),
                ('Total Words Generated', f"{bert_df['word_count'].sum():,}")
            ]
        }
        
        # Save basic statistics
        for category, stats in basic_stats.items():
            stats_df = pd.DataFrame(stats, columns=['Metric', 'Value'])
            stats_path = os.path.join(self.dirs['statistics'], f'{category.lower().replace(" ", "_")}.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"  ✓ {category} saved: {stats_path}")
        
        # 2. Generate visualizations
        self._create_visualizations(clinical_df, bert_df)
        
        # 3. Field completeness analysis
        completeness_df = pd.DataFrame([
            {'Field': col, 'Non-Null': count, 'Total': self.stats['total_patients'],
             'Completeness': f"{count/self.stats['total_patients']*100:.1f}%"}
            for col, count in self.stats['field_completeness'].items()
        ]).sort_values('Completeness', ascending=False)
        
        completeness_path = os.path.join(self.dirs['statistics'], 'field_completeness.csv')
        completeness_df.to_csv(completeness_path, index=False)
        
        print(f"  ✓ Field completeness analysis: {completeness_path}")
    
    def _create_visualizations(self, clinical_df: pd.DataFrame, bert_df: pd.DataFrame):
        """Create visualization plots"""
        
        print("\nCreating visualizations...")
        
        try:
            # 1. Text Length Distribution
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Clinical Report Generation Analysis', fontsize=16, fontweight='bold')
            
            # 1a. BERT text length distribution
            ax = axes[0, 0]
            ax.hist(bert_df['text_length'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Text Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('BERT Text Length Distribution')
            ax.axvline(bert_df['text_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {bert_df["text_length"].mean():.0f}')
            ax.legend()
            
            # 1b. Word count distribution
            ax = axes[0, 1]
            ax.hist(bert_df['word_count'], bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Word Count Distribution')
            ax.axvline(bert_df['word_count'].mean(), color='red', linestyle='--',
                      label=f'Mean: {bert_df["word_count"].mean():.0f}')
            ax.legend()
            
            # 1c. pCR distribution
            ax = axes[0, 2]
            pcr_data = self.stats['pcr_distribution']
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax.pie(pcr_data.values(), labels=pcr_data.keys(), autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title('Pathological Complete Response (pCR) Distribution')
            
            # 1d. Subtype distribution
            ax = axes[1, 0]
            subtype_data = self.stats['subtype_distribution']
            if subtype_data:
                subtypes = list(subtype_data.keys())
                counts = list(subtype_data.values())
                bars = ax.barh(subtypes, counts, color='steelblue')
                ax.set_xlabel('Number of Patients')
                ax.set_title('Tumor Subtype Distribution')
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.0f}', va='center')
            
            # 1e. Therapy distribution
            ax = axes[1, 1]
            therapy_data = self.stats['therapy_distribution']
            if therapy_data:
                therapies = list(therapy_data.keys())
                counts = list(therapy_data.values())
                bars = ax.bar(therapies, counts, color=['#3498db', '#9b59b6', '#1abc9c'])
                ax.set_ylabel('Number of Patients')
                ax.set_title('Therapy Distribution')
                ax.set_xticklabels(therapies, rotation=45, ha='right')
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.01,
                           f'{height:.0f}', ha='center', va='bottom')
            
            # 1f. Field completeness (top 10)
            ax = axes[1, 2]
            completeness = {
                col: count/self.stats['total_patients']*100 
                for col, count in self.stats['field_completeness'].items()
            }
            top_fields = dict(sorted(completeness.items(), key=lambda x: x[1], reverse=True)[:10])
            bars = ax.barh(list(top_fields.keys()), list(top_fields.values()), color='coral')
            ax.set_xlabel('Completeness (%)')
            ax.set_title('Top 10 Most Complete Fields')
            ax.set_xlim(0, 100)
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%', va='center')
            
            plt.tight_layout()
            plot_path = os.path.join(self.dirs['visualizations'], 'generation_summary.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Summary visualization: {plot_path}")
            
            # 2. Text Length vs Word Count scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(bert_df['text_length'], bert_df['word_count'], alpha=0.6, 
                       c=bert_df['has_error'].map({True: 'red', False: 'blue'}))
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Word Count')
            plt.title('BERT Text Length vs Word Count')
            
            # Add regression line for non-error points
            non_error = bert_df[~bert_df['has_error']]
            if len(non_error) > 1:
                z = np.polyfit(non_error['text_length'], non_error['word_count'], 1)
                p = np.poly1d(z)
                plt.plot(non_error['text_length'], p(non_error['text_length']), 
                        "r--", alpha=0.8, label=f'Correlation: {np.corrcoef(non_error["text_length"], non_error["word_count"])[0,1]:.3f}')
                plt.legend()
            
            plot_path = os.path.join(self.dirs['visualizations'], 'text_length_vs_word_count.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Text analysis plot: {plot_path}")
            
            # 3. Save data for plots as CSV (for LaTeX/paper)
            plot_data = {
                'text_length_stats': {
                    'mean': float(bert_df['text_length'].mean()),
                    'std': float(bert_df['text_length'].std()),
                    'min': float(bert_df['text_length'].min()),
                    'max': float(bert_df['text_length'].max())
                },
                'pcr_distribution': self.stats['pcr_distribution'],
                'subtype_distribution': self.stats['subtype_distribution'],
                'top_complete_fields': dict(list(
                    sorted(self.stats['field_completeness'].items(), 
                          key=lambda x: x[1]/self.stats['total_patients'], 
                          reverse=True)[:5]
                ))
            }
            
            plot_data_path = os.path.join(self.dirs['visualizations'], 'plot_data.json')
            with open(plot_data_path, 'w') as f:
                json.dump(plot_data, f, indent=2)
            
            # Also save as CSV for tables
            for data_name, data in plot_data.items():
                if isinstance(data, dict):
                    df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
                    df.to_csv(os.path.join(self.dirs['visualizations'], f'{data_name}.csv'), index=False)
            
            print(f"  ✓ Plot data saved: {plot_data_path}")
            
        except Exception as e:
            print(f"  ⚠ Warning: Error creating visualizations: {e}")
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper inclusion"""
        
        print("\nGenerating LaTeX tables for paper...")
        
        # Table 1: Dataset characteristics
        table1_data = [
            ("Total Patients", f"{self.stats['total_patients']}"),
            ("Successful Reports", f"{self.stats['successful_generations']}"),
            ("Success Rate", f"{self.stats['successful_generations']/self.stats['total_patients']*100:.1f}\%"),
            ("Average BERT Text Length", f"{np.mean(self.stats['text_lengths']):.0f} characters"),
            ("Total Words Generated", f"{sum([len(t.split()) for t in self.stats['text_lengths']]):,}")
        ]
        
        with open(os.path.join(self.dirs['latex_tables'], 'table1_dataset.tex'), 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Dataset characteristics and report generation statistics}\n")
            f.write("\\label{tab:dataset_stats}\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Characteristic & Value \\\\\n")
            f.write("\\midrule\n")
            for name, value in table1_data:
                f.write(f"{name} & {value} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        # Table 2: Clinical characteristics distribution
        table2_data = []
        for subtype, count in self.stats['subtype_distribution'].items():
            percentage = count / self.stats['total_patients'] * 100
            table2_data.append((subtype, f"{count}", f"{percentage:.1f}\%"))
        
        with open(os.path.join(self.dirs['latex_tables'], 'table2_clinical.tex'), 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Distribution of tumor molecular subtypes}\n")
            f.write("\\label{tab:subtype_distribution}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\toprule\n")
            f.write("Molecular Subtype & Count & Percentage \\\\\n")
            f.write("\\midrule\n")
            for subtype, count, perc in table2_data:
                f.write(f"{subtype} & {count} & {perc} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        # Table 3: pCR distribution
        table3_data = []
        for status, count in self.stats['pcr_distribution'].items():
            percentage = count / self.stats['total_patients'] * 100
            status_name = status.replace('_', ' ').title()
            table3_data.append((status_name, f"{count}", f"{percentage:.1f}\%"))
        
        with open(os.path.join(self.dirs['latex_tables'], 'table3_pcr.tex'), 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Distribution of pathological complete response (pCR)}\n")
            f.write("\\label{tab:pcr_distribution}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\toprule\n")
            f.write("pCR Status & Count & Percentage \\\\\n")
            f.write("\\midrule\n")
            for status, count, perc in table3_data:
                f.write(f"{status} & {count} & {perc} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"  ✓ LaTeX tables saved to: {self.dirs['latex_tables']}/")
        print(f"    1. table1_dataset.tex - Dataset characteristics")
        print(f"    2. table2_clinical.tex - Subtype distribution")
        print(f"    3. table3_pcr.tex - pCR distribution")
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        
        print("\n" + "="*80)
        print("CLINICAL REPORT GENERATION - COMPREHENSIVE SUMMARY")
        print("="*80)
        
        print(f"\n📊 GENERATION STATISTICS")
        print("-" * 40)
        print(f"Total Patients Processed: {self.stats['total_patients']}")
        print(f"Successful Report Generations: {self.stats['successful_generations']}")
        print(f"Failed Report Generations: {self.stats['failed_generations']}")
        success_rate = self.stats['successful_generations'] / self.stats['total_patients'] * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n📝 TEXT STATISTICS")
        print("-" * 40)
        if self.stats['text_lengths']:
            print(f"Average BERT Text Length: {np.mean(self.stats['text_lengths']):.0f} characters")
            print(f"Text Length Range: {min(self.stats['text_lengths']):.0f} - {max(self.stats['text_lengths']):.0f} characters")
        
        print(f"\n🏥 CLINICAL CHARACTERISTICS")
        print("-" * 40)
        print("pCR Distribution:")
        for status, count in self.stats['pcr_distribution'].items():
            percentage = count / self.stats['total_patients'] * 100
            print(f"  {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print("\nTumor Subtype Distribution:")
        for subtype, count in self.stats['subtype_distribution'].items():
            percentage = count / self.stats['total_patients'] * 100
            print(f"  {subtype}: {count} ({percentage:.1f}%)")
        
        print(f"\n📁 OUTPUT DIRECTORY STRUCTURE")
        print("-" * 40)
        print(f"Root: {self.output_base_dir}")
        for dir_name, dir_path in self.dirs.items():
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                print(f"  {dir_name}/ - {file_count} files")
        
        print(f"\n📍 ALL FILES SAVED TO: {self.output_base_dir}")
        print("="*80)

def load_and_clean_data(file_path: str, sheet_name: str = 'dataset_info') -> pd.DataFrame:
    """Load and clean the clinical data"""
    
    print(f"Loading data from: {file_path}")
    print(f"Sheet name: {sheet_name}")
    
    try:
        # Try to read Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✓ Successfully loaded {len(df)} records")
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("\nAvailable sheets:")
        try:
            xl = pd.ExcelFile(file_path)
            print(f"  {xl.sheet_names}")
            # Try first sheet if specified sheet not found
            if sheet_name not in xl.sheet_names:
                print(f"Sheet '{sheet_name}' not found. Using first sheet: {xl.sheet_names[0]}")
                df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])
        except Exception as e2:
            print(f"  Cannot read Excel file: {e2}")
            raise
    
    # Clean column names
    print("\nCleaning column names...")
    original_cols = df.columns.tolist()
    
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('(', '')
        .str.replace(')', '')
        .str.replace('-', '_')
        .str.replace('/', '_')
        .str.replace('\\', '_')
    )
    
    # Show column mapping
    print("Column name mapping (changed names):")
    changes = 0
    for orig, new in zip(original_cols, df.columns):
        if orig != new:
            print(f"  '{orig}' -> '{new}'")
            changes += 1
    
    if changes == 0:
        print("  (No changes needed)")
    
    # Check required columns
    print("\nChecking data quality...")
    required_cols = ['patient_id']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠ Warning: Missing required columns: {missing}")
        # Try to find similar columns
        for miss in missing:
            similar = [col for col in df.columns if miss.lower() in col.lower()]
            if similar:
                print(f"  Similar columns found for '{miss}': {similar}")
    
    # Show data preview
    print(f"\nData preview (shape: {df.shape}):")
    print(df.head(3))
    
    # Show missing values
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0]
    if len(missing_cols) > 0:
        print(f"\nMissing values per column (columns with missing values only):")
        for col, missing_count in missing_cols.items():
            percentage = missing_count / len(df) * 100
            print(f"  {col:25s}: {missing_count:4d} missing ({percentage:5.1f}%)")
    else:
        print("\nNo missing values in any column!")
    
    return df

def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = r'D:\MAMA-MIA\clinical_and_imaging_info.xlsx'
    SHEET_NAME = 'dataset_info'
    
    print("="*80)
    print("CLINICAL REPORT GENERATOR - FIXED VERSION")
    print("="*80)
    print(f"Input file: {DATA_PATH}")
    print("="*80)
    
    try:
        # Step 1: Load and clean data
        print("\n[STEP 1] LOADING AND CLEANING DATA")
        print("-" * 40)
        df = load_and_clean_data(DATA_PATH, SHEET_NAME)
        
        # Step 2: Initialize generator (auto-creates timestamped directory)
        print("\n[STEP 2] INITIALIZING REPORT GENERATOR")
        print("-" * 40)
        generator = ClinicalReportGenerator()  # Auto creates timestamped directory
        
        # Step 3: Generate reports
        print("\n[STEP 3] GENERATING REPORTS")
        print("-" * 40)
        clinical_df, bert_df = generator.process_dataframe(df)
        
        # Step 4: Save outputs
        print("\n[STEP 4] SAVING OUTPUTS")
        print("-" * 40)
        clinical_path, bert_path, corpus_path = generator.save_outputs(clinical_df, bert_df)
        
        # Step 5: Generate statistics and visualizations
        print("\n[STEP 5] GENERATING STATISTICS & VISUALIZATIONS")
        print("-" * 40)
        generator.generate_statistics_and_plots(clinical_df, bert_df)
        
        # Step 6: Generate LaTeX tables
        print("\n[STEP 6] GENERATING LATEX TABLES FOR PAPER")
        print("-" * 40)
        generator.generate_latex_tables()
        
        # Step 7: Print summary
        print("\n[STEP 7] FINAL SUMMARY")
        print("-" * 40)
        generator.print_summary_report()
        
        print(f"\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
        print(f"📁 All outputs saved to: {generator.output_base_dir}")
        
        # Show exact file paths for easy access
        print("\n📋 KEY OUTPUT FILES:")
        print(f"  1. Structured Reports: {os.path.join(generator.dirs['reports'], 'clinical_structured_reports.csv')}")
        print(f"  2. BERT Texts Index: {os.path.join(generator.dirs['bert_texts'], 'bert_input_texts.csv')}")
        print(f"  3. BERT Corpus: {os.path.join(generator.dirs['bert_texts'], 'bert_corpus.txt')}")
        print(f"  4. Individual BERT Files: {generator.dirs['bert_individual']}/*.txt")
        print(f"  5. Visualizations: {generator.dirs['visualizations']}/*.png")
        print(f"  6. Statistics: {generator.dirs['statistics']}/*.csv")
        print(f"  7. LaTeX Tables: {generator.dirs['latex_tables']}/*.tex")
        print(f"  8. Metadata: {os.path.join(generator.dirs['metadata'], 'generation_metadata.json')}")
        
        return 0
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at {DATA_PATH}")
        print("Please check the file path and try again.")
        return 1
    except PermissionError:
        print(f"\n❌ ERROR: Permission denied when trying to access {DATA_PATH}")
        print("Please check file permissions or close the file if it's open in another program.")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR in main execution: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run the main function
    print("Starting clinical report generation...")
    print("This will create a new timestamped directory for all outputs.")
    print()
    
    success = main()
    
    if success == 0:
        print("\n" + "="*80)
        print("🎉 GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps for BERT feature extraction:")
        print("1. Load BERT texts from the generated CSV file")
        print("2. Use individual .txt files in bert_individual_files/")
        print("3. Use the corpus file for pretraining")
        print("\nFor your paper, use:")
        print("  • PNG files in visualizations/ for figures")
        print("  • LaTeX files in latex_tables/ for tables")
        print("  • CSV files in statistics/ for data")
    else:
        print("\n❌ Generation failed. Please check the error messages above.")
    
    # Keep window open
    input("\nPress Enter to exit...")