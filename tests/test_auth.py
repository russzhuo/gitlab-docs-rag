from app.services.auth_service import hash_password, verify_password


def test_password_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)


def test_password_wrong_fails():
    h = hash_password("secret123")
    assert not verify_password("wrong", h)
