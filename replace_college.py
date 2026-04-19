import os

replacements = [
    ("S. B. Patil Educational Campus", "Sharadchandra Pawar College of Engineering & Technology"),
    ("S. B. Patil Educational <br /> Campus, Pune", "Sharadchandra Pawar College of <br /> Engineering & Technology"),
    ("SBPEC", "SPCET"),
    ("Shahajirao Patil Vikas Pratishthan", "Shri Someshwar Shikshan Prasarak Mandal's"),
    ("Vangali, Indapur, Pune - 413106. Landmark: SBP Campus.", "Someshwarnagar, Tal - Baramati, Dist - Pune 412306"),
    ("Vangali, Indapur, Pune - 413106", "Someshwarnagar, Tal - Baramati, Dist- Pune 412306"),
    ("020-24354705", "(02112) 283185"),
    ("principal.scoe@dtu.ac.in", "sspm1972@gmail.com"),
    ("Approved by AICTE New Delhi, Affiliated to Savitribai Phule Pune University (SPPU)", "Approved by AICTE New Delhi, Recognized by Govt. of Maharashtra & Affiliated to University of Pune"),
    ("Recognised by Govt. of Maharashtra, DTE Mumbai.", "Id.no.PU/PN.Engg./445/2012"),
    ("DTE Code 6177 | NAAC : A+ Grade | NBA Accredited | ISO 9001: 2015", ""),
    ("Govt. of NCT of Delhi's", "Shri Someshwar Shikshan Prasarak Mandal's"),
    ("NTC, Delhi", "Pune"),
    ("Indapur, Pune", "Baramati, Pune"),
]

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith(('.tsx', '.ts')):
            process_file(os.path.join(root, file))

if os.path.exists('README.md'):
    process_file('README.md')
