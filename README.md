# Patient Record System (PRS)

A Python web application for managing synthetic patient records and dated visit notes, with staff-reviewed AI categorization. Built with Flask, SQLite, and object-oriented Python. **For educational demonstration with synthetic patient information only; not for clinical use.**

## Requirements

- Python 3.10 or newer (Python 3.12 recommended on Windows)
- Internet access for the first download of the pretrained AI model

## Install and run (Windows / VS Code)

1. Extract this ZIP and open the folder containing `app.py` in VS Code.
2. Open a terminal in that folder and run:

   ```powershell
   py -m pip install -r requirements.txt
   py -m pip install -r requirements-ai.txt
   py app.py
   ```

3. Open **http://127.0.0.1:5000** in your browser.
4. Create a manager account (minimum 8-character password), then sign in.
5. Register a synthetic patient, locate the record by Patient ID, add a dated visit note, and review any suggested category.

If your computer uses `python` rather than `py`, use `python` for each command. To stop the server, press **Ctrl+C** in its terminal.

## Main features

- **Patient management:** Register synthetic patients with unique immutable Patient IDs; search and update permitted fields.
- **Visit notes:** Add dated notes to existing patients and retrieve their visit history. Saved note contents cannot be altered or deleted through the application.
- **Authentication and permissions:** Manager, staff, and doctor accounts with different permissions.
- **AI-assisted categorization:** A pretrained local zero-shot classification model suggests *Respiratory*, *Digestive*, *Musculoskeletal*, or *General/Other*. A staff member can accept, reject, or leave a suggestion pending. The AI never diagnoses patients, and note saving continues if the model is missing, unavailable, or uncertain.
- **Persistent storage:** SQLite writes records to `prs.db` and keeps them after the program restarts.

## AI assistance

The AI uses the pretrained `cross-encoder/nli-MiniLM2-L6-H768` model through PyTorch and Transformers. The model files are downloaded when the first visit note is submitted or the AI Status test is run. This may take time depending on the network and machine. The AI uses relative classification scores rather than clinically calibrated probabilities, and suggestions below the configured threshold are withheld.

After signing in, open **AI status → Test AI with sample note**. This uses a predefined synthetic note and displays model availability, suggested category, and category scores. If the AI is unavailable, the page shows a diagnostic reason. See `AI_SETUP_WINDOWS.txt` for troubleshooting.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask routes and web interface |
| `prs.py` | Object-oriented patient, visit, authentication, database, and AI services |
| `templates/` | HTML page templates |
| `static/` | CSS styles |
| `requirements.txt` | Web dependencies |
| `requirements-ai.txt` | Optional pretrained-model dependencies |
| `AI_SETUP_WINDOWS.txt` | AI installation and troubleshooting instructions |
| `test_core.py`, `test_ai.py` | Automated checks |

## Storage and safety

The database file `prs.db` is created in the working folder. Stop the server before copying the file to another folder. The application is intended for local coursework demonstrations, not internet deployment or real patient data.
