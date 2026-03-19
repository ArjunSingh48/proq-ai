How to run:

## Local development

Frontend and backend are executed separately (in 2 separate shells):
1. Frontend:
  ```
  cd frontend
  npm run dev
  ```

2. Backend:
   ```
   python -m uvicorn app:app --reload --port 8000 --log-level debug
   ```

## Vercel deployment

The project is configured for Vercel:
- Frontend (Vite) is built from `frontend/` and served as static files
- Backend (FastAPI) runs as a serverless function via `api/index.py`

To deploy:
```
vercel
```
