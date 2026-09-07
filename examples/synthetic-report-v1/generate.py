"""Generate a deterministic, one-page, synthetic text PDF using only stdlib."""
from pathlib import Path

LINES = [
    'Quarterly Sales Report', '',
    'Region      Q1 Revenue   Q2 Revenue',
    'North       120,000      135,000',
    'South       98,500       102,300',
    'East        87,200       91,000', '',
    'Prepared for internal review.',
]


def main():
    output = Path(__file__).parent / 'input' / 'sample_report.pdf'
    output.parent.mkdir(exist_ok=True)
    commands = ['BT', '/F1 12 Tf', '50 760 Td', '18 TL']
    for index, line in enumerate(LINES):
        if index:
            commands.append('T*')
        escaped = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        commands.append(f'({escaped}) Tj')
    stream = ('\n'.join(commands + ['ET']) + '\n').encode('ascii')
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>',
        f'<< /Length {len(stream)} >>\nstream\n'.encode() + stream + b'endstream',
    ]
    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f'{number} 0 obj\n'.encode() + obj + b'\nendobj\n')
    xref = len(pdf)
    pdf.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode())
    pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    output.write_bytes(pdf)
    print(output)


if __name__ == '__main__':
    main()
