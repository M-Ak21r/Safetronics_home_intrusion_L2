# Authorized Personnel Directory

This directory contains images of authorized personnel for the SENTINEL Guardian system.

## How to Add Authorized Personnel

1. Place a clear, front-facing photo of each authorized person in this directory.
2. Name the file with the person's name (this will be displayed when they are recognized).
3. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Example

```
authorized_personels/
├── README.md
├── john_smith.jpg
├── jane_doe.png
└── security_guard.jpg
```

## Photo Requirements

- Clear, well-lit front-facing photo
- Single face per image
- Minimum resolution: 128x128 pixels
- Recommended resolution: 400x400 pixels or higher

## Notes

- The system will extract the person's name from the filename (without extension)
- Multiple images of the same person can be added with different suffixes (e.g., `john_smith_1.jpg`, `john_smith_2.jpg`)
- If no face is detected in an image, it will be skipped with a warning
