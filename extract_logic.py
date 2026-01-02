import json
import sys
import re

# Force utf-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\notebooks\structuredaccidentseveritymlnotebook.ipynb"
output_file = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\analysis_output.txt"

def analyze_notebook():
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as out:
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    source = "".join(cell['source'])
                    
                    # 1. capture Weather_Simplified mapping
                    if "Weather_Simplified" in source and ("map" in source or "apply" in source or "def " in source):
                        out.write(f"--- Cell {i} (Weather Mapping) ---\n")
                        out.write(source)
                        out.write("\n" + "-" * 20 + "\n")

                    # 2. capture Feature Selection / Drop
                    if "drop" in source and "columns" in source:
                         out.write(f"--- Cell {i} (Col Drop) ---\n")
                         out.write(source)
                         out.write("\n" + "-" * 20 + "\n")

                    # 3. capture LGBM training columns
                    if "lgbm" in source.lower() and ("fit" in source or "train" in source):
                         out.write(f"--- Cell {i} (LGBM Fit) ---\n")
                         out.write(source)
                         out.write("\n" + "-" * 20 + "\n")
                         
                    # 4. Look for X_train definition to see columns
                    if "X_train" in source and "=" in source:
                        out.write(f"--- Cell {i} (X_train def) ---\n")
                        out.write(source[:1000]) 
                        out.write("\n" + "-" * 20 + "\n")

        print(f"Analysis written to {output_file}")

    except Exception as e:
        print(f"Error processing notebook: {e}")

if __name__ == "__main__":
    analyze_notebook()
