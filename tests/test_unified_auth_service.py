import warnings
import pytest
from bson import ObjectId

from arionxiv.services.unified_auth_service import UnifiedAuthenticationService


warnings.filterwarnings("ignore", message="PyPDF2 is deprecated", category=DeprecationWarning)

class _StubDBService:
    def __init__(self, user_doc):
        self._user_doc = user_doc
        self.updated = False

    async def find_one(self, collection, filter_dict):
        return self._user_doc

    async def update_one(self, collection, filter_dict, update_dict):
        self.updated = True
        class Result:
            modified_count = 1
        return Result()


@pytest.mark.asyncio
async def test_password_login_rejected_for_non_local(monkeypatch):
    service = UnifiedAuthenticationService()
    user_doc = {
        '_id': ObjectId(),
        'email': 'user@example.com',
        'is_active': True,
        'auth_provider': 'external'
    }
    stub_db = _StubDBService(user_doc)
    monkeypatch.setattr('arionxiv.services.unified_auth_service.unified_database_service', stub_db)

    result = await service.authenticate_user('user@example.com', 'irrelevant')

    assert result['success'] is False
    assert 'linked provider' in result['error']
    assert stub_db.updated is False
