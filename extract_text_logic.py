import json
import sys

# Force utf-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\notebooks\structuredaccidentseveritymlnotebook.ipynb"
output_file = r"c:\Users\nwagb\Desktop\MACHINE_LEARNING_ASSESSEMENT\us_accident_prediction_model\text_features.txt"

def analyze_text_features():
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as out:
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    source = "".join(cell['source'])
                    
                    # Capture creation of Desc_Queue or road work logic
                    if "Description" in source and ("Queue" in source or "Road Work" in source or "str.contains" in source):
                        out.write(f"--- Cell {i} (Text Features) ---\n")
                        out.write(source)
                        out.write("\n" + "-" * 20 + "\n")
                        
                    # Check for Start_Lat drops again specifically
                    if "drop" in source and "Start_Lat" in source:
                         out.write(f"--- Cell {i} (Start_Lat Drop) ---\n")
                         out.write(source)
                         out.write("\n" + "-" * 20 + "\n")

        print(f"Text analysis written to {output_file}")

    except Exception as e:
        print(f"Error processing notebook: {e}")

if __name__ == "__main__":
    analyze_text_features()
