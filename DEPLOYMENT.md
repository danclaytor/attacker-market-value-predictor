# Deploy to Streamlit Community Cloud

## Deployment status

Live app: https://danclaytor-player-values.streamlit.app/

Repository: https://github.com/danclaytor/attacker-market-value-predictor

Deployment: `main` branch, `app.py`, Python 3.12.

## Publish the repository

Create a public GitHub repository named `attacker-market-value-predictor` (or choose another name). Upload the project with its directory structure intact, including `data/`, `preparation/`, `.streamlit/config.toml`, `requirements.txt` and `app.py`. Do not upload `.venv/`, caches, raw downloads, `.env` files or credentials.

## Deploy

1. Sign in at https://share.streamlit.io/ using your GitHub account and authorise the required repository access.
2. Select **Create app**, then the option to deploy an app from GitHub.
3. Choose the published repository, branch `main`, and main file `app.py`.
4. In advanced settings, select **Python 3.12**. No secrets are required.
5. Choose an available app subdomain and deploy.
6. Confirm the app's sharing settings allow anyone to view it.
7. Open the public URL in a signed-out/private browser and verify player selection, hypothetical inputs, reset, charts and CSV download.
8. Put the verified URL in the README and the repository's **About → Website** field. Replace this pending status with the live URL.

The repository hosts the code; visitors use the deployed `streamlit.app` URL. GitHub Pages alone cannot run this Python server.

## Maintain

Push changes to the deployed branch to update the app. Keep runtime packages in `requirements.txt`; optional research/report tooling lives in `requirements-analysis.txt`. Use the bundled data to avoid live downloads during app startup. Run `python -m pytest -q` before publishing changes. A sleeping Community Cloud app may take time to wake on the next visit.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

