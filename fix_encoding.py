filepath = 'materi-pendat/UTS_Analisa_Kesuburan_Tanah.md'

with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix double CRLF -> LF
content = content.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')

# Fix specific garbled patterns in lines 41 and 61
# These are garbled em-dash sequences
import re

# Pattern: garbled em dash (multiple variants due to double encoding)
garbled_patterns = [
    r'ÃÂ¢Ã¢âÂ¬"Â',  # worst case triple encoding
    r'Ã¢â‚¬â€"',
    r'â€"',
    r'Ã¢â‚¬â€œ',
    r'â€"',
    r'â€"',
    r'Â³',
    r'Ã‚Â³',
]

for pat in garbled_patterns:
    if pat in content:
        print(f'Replacing: {pat!r}')
        content = content.replace(pat, '-')

# Also fix cm3 specifically
content = content.replace('g/cm-', 'g/cm3')
content = content.replace('cm-', 'cm3')

# Check remaining non-ASCII outside normal range
issues = 0
for i, c in enumerate(content):
    code = ord(c)
    if code > 127 and code not in [178, 179, 185, 176, 177, 215, 247, 178, 179,
                                    # Greek letters
                                    945,946,947,948,949,950,951,952,
                                    # arrows
                                    8594,8592,8593,8595,
                                    # math
                                    8804,8805,8800,8721,8747,8730,8734,960,956,963,961,964,
                                    # superscripts
                                    179,178,185,
                                    # regular extended latin
                                    233,232,234,235,224,225,226,227,228,229,
                                    230,231,236,237,238,239,242,243,244,245,
                                    246,249,250,251,252,253,254,192,193,194,
                                    195,196,197,198,199,200,201,202,203,204,
                                    205,206,207,208,209,210,211,212,213,214,
                                    216,217,218,219,220,221,222,223,255]:
        ctx = content[max(0,i-15):i+15]
        print(f'Non-standard char at {i}: ord={code}, hex={hex(code)}, ctx={ctx!r}')
        issues += 1
        if issues > 10:
            print('...more issues exist')
            break

with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print(f'\nDone! Found {issues} remaining issues')
