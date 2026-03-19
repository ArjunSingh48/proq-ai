How to run:

Frontend and backend are executed separately (in 2 separate shells):
1. Frontend:
  ```
cd frontend
npm run dev
  ```

2. Backend:
   ```
   python -m uvicorn fastapi_app:app --reload --port 8000 --log-level debug
   ```
