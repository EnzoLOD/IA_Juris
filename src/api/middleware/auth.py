from fastapi import Request, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.config.settings import SECRET_KEY
from src.utils.exceptions import InvalidCredentialsException
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = payload.get("sub")
        if user is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return user

async def authenticate_user(form_data: OAuth2PasswordRequestForm):
    user = await get_user(form_data.username)
    if not user or not verify_password(form_data.password, user['hashed_password']):
        raise InvalidCredentialsException
    return user