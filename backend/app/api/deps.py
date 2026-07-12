from app.services.session import SessionStore, session_store


def get_session_store() -> SessionStore:
    return session_store
