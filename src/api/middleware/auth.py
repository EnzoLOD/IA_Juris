import logging
from flask import Blueprint, request, jsonify, current_app, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user, fresh_login_required
from werkzeug.security import generate_password_hash # Mantido para referência, bcrypt.hashpw é usado
import bcrypt # Biblioteca principal para hashing de senhas
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError # Para tratamento de erros de banco de dados

# --- Placeholder Imports ---
# IMPORTANT: Substitua 'yourapplication' pelo nome real do seu pacote Flask.
# Por exemplo, se seu app está em 'my_app', use 'from my_app import db, models, forms'.
try:
    from yourapplication import db # Seu objeto SQLAlchemy, inicializado em __init__.py
    from yourapplication.models import User # Seu modelo de usuário (deve herdar de UserMixin e db.Model)
    from yourapplication.forms import LoginForm, RegistrationForm # Seus formulários WTForms
    from flask_wtf.csrf import CSRFProtect # Para proteção CSRF em APIs ou quando não usar WTForms diretamente
    # Instância global de CSRFProtect. Será inicializada com o app em init_auth.
    csrf = CSRFProtect() 
except ImportError as e:
    # Este bloco permite que o arquivo seja inspecionado mesmo sem a estrutura completa da app.
    # Em um ambiente de produção, certifique-se de que esses imports funcionem.
    print(f"ATENÇÃO: Não foi possível importar classes da sua aplicação ('yourapplication'). Usando Dummies para demonstração. Erro: {e}")

    # Dummy classes para permitir que o código seja parseado e inspecionado
    # NÃO USE ESTAS CLASSES EM PRODUÇÃO!
    class DummyDB:
        def __init__(self):
            self.session = self
        def add(self, obj): pass
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass # Adiciona close para simular db.session.close()
    db = DummyDB()

    class DummyUserMixin:
        is_active = True
        is_authenticated = True
        is_anonymous = False
        # Para Flask-Login, get_id() deve retornar uma string
        def get_id(self): return str(self.id)
        # Adicione uma propriedade 'email' para que o logger não falhe
        @property
        def email(self): return "dummy@example.com"
        def is_fresh(self): return True # Simula um login "fresco"

    class DummyUser(DummyUserMixin):
        def __init__(self, id=1, email="dummy@example.com", password="hashed_password"):
            self.id = id
            self.email = email
            self.password = password
        def __repr__(self): return f"DummyUser('{self.email}')"
        # Adiciona métodos de senha para compatibilidade com o exemplo de modelo
        def set_password(self, password):
            self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        def check_password(self, password):
            # Para o dummy, apenas verifica se a senha não está vazia
            return bool(password)

    class DummyForm:
        def __init__(self, data=None, meta=None, **kwargs):
            self.data = data or {}
            self.email = type('obj', (object,), {'data': self.data.get('email')})()
            self.password = type('obj', (object,), {'data': self.data.get('password')})()
            self.remember_me = type('obj', (object,), {'data': self.data.get('remember_me', False)})()
            self.confirm_password = type('obj', (object,), {'data': self.data.get('confirm_password')})()
        def validate_on_submit(self):
            # Para dummies, sempre valida se houver email e senha
            return bool(self.data.get('email') and self.data.get('password'))
        @property
        def errors(self):
            return {"_form": ["Dummy form validation failed."]} # Exemplo de erro dummy
    LoginForm = DummyForm
    RegistrationForm = DummyForm

    class DummyCSRFProtect:
        def init_app(self, app): pass
        def protect(self): pass
        def exempt(self, f): return f
        def generate_csrf(self): return "dummy_csrf_token"
        def get_csrf_token(self): return "dummy_csrf_token"
        def error_handler(self, f): return f
    csrf = DummyCSRFProtect()


# --- Configuração de Logging ---
# Configura o logger para exibir mensagens informativas e de depuração.
# É crucial para monitorar tentativas de login, registros e erros de segurança.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Inicialização do Blueprint e CSRFProtect ---
auth = Blueprint('auth', __name__)
# CSRFProtect será inicializado com o objeto 'app' na função init_auth()

# --- Configuração do Flask-Login ---
login_manager = LoginManager()
login_manager.login_view = 'auth.login' # Endpoint para redirecionar usuários não autenticados
login_manager.login_message = "Por favor, faça login para acessar esta página."
login_manager.login_message_category = "info" # Categoria para mensagens flash
login_manager.needs_refresh_message = (
    "Sua sessão expirou ou não é 'fresh'. Por favor, faça login novamente para acessar esta funcionalidade."
)
login_manager.needs_refresh_message_category = "warning" # Categoria para mensagens flash

# Handler de erro para CSRF. Retorna JSON para APIs.
@csrf.error_handler
def csrf_error(reason):
    logger.warning(f"CSRF token validation failed: {reason}")
    return jsonify({'message': 'CSRF token missing or invalid', 'reason': reason}), 403

@login_manager.user_loader
def load_user(user_id):
    """
    Função callback do Flask-Login para carregar um usuário pelo ID.
    Essencial para manter o estado da sessão do usuário entre requisições.
    O `user_id` é o valor retornado por `User.get_id()`.
    """
    if user_id is None:
        return None
    try:
        # Tenta converter user_id para int. Flask-Login armazena como string.
        user = User.query.get(int(user_id))
        logger.debug(f"User loaded: {user.email if user else 'None'} for ID: {user_id}")
        return user
    except ValueError:
        logger.error(f"Invalid user_id format provided to load_user: {user_id}. Expected integer.", exc_info=True)
        return None
    except OperationalError as e:
        logger.error(f"Database operational error when loading user ID {user_id}: {e}", exc_info=True)
        # Em caso de erro de DB, o usuário não pode ser carregado.
        # Considere redirecionar para uma página de erro ou deslogar o usuário.
        return None
    except SQLAlchemyError as e:
        logger.error(f"A SQLAlchemy error occurred while loading user ID {user_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading user ID {user_id}: {e}", exc_info=True)
        return None

# --- Rotas de Autenticação ---

@auth.route('/login', methods=['POST'])
# Para endpoints de API que recebem JSON, Flask-WTF pode precisar que você use @csrf.protect
# Se o formulário (FlaskForm) é usado, `form.validate_on_submit()` já faz a validação CSRF
# através de um campo oculto `csrf_token` no formulário HTML.
# Se você está recebendo JSON e não está usando um `FlaskForm` diretamente para validação
# (ex: apenas `request.get_json()`), você precisaria de `@csrf.protect` para forçar a verificação do token
# no cabeçalho `X-CSRFToken`. No entanto, como estamos usando `LoginForm`, o `validate_on_submit`
# já cuida disso para ambos os casos (form HTML ou JSON com campo CSRF).
# Portanto, `@csrf.exempt` não é estritamente necessário se `validate_on_submit` é o único ponto de validação CSRF.
# Contudo, mantê-lo aqui não causa problemas e pode ser útil se a lógica de validação mudar.
def login():
    """
    Endpoint para login de usuário.
    Espera email e senha. Suporta tanto `request.form` (HTML forms) quanto `request.get_json()` (APIs JSON).
    """
    # Determina se a requisição é JSON ou um formulário padrão.
    # Isso permite que o mesmo endpoint sirva tanto a clientes web (HTML forms) quanto a APIs (JSON).
    data = request.get_json() if request.is_json else request.form

    # Inicializa o formulário com os dados recebidos.
    # Flask-WTF pode extrair dados de `request.form` ou `request.get_json()`.
    form = LoginForm(data=data)

    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        try:
            user = User.query.filter_by(email=email).first()

            # Verifica se o usuário existe e a senha está correta.
            # É crucial que `user.password` seja o hash da senha armazenado
            # e que `password.encode('utf-8')` seja a senha bruta para comparação.
            # O método `check_password` encapsula a lógica do bcrypt.
            if user and user.check_password(password):
                # Autenticação bem-sucedida.
                # `remember_me` permite que o cookie de sessão persista após o fechamento do navegador.
                remember_me = form.remember_me.data if hasattr(form, 'remember_me') else False
                login_user(user, remember=remember_me)
                logger.info(f"User {email} logged in successfully.")
                
                # Resposta JSON para APIs.
                # Para aplicações web tradicionais, você usaria `redirect(url_for('main.dashboard'))` e `flash()`.
                return jsonify({'message': 'Logged in successfully', 'user_id': user.id, 'email': user.email}), 200
            else:
                # Credenciais inválidas (usuário não encontrado ou senha incorreta).
                # Evita detalhar o motivo exato (usuário/senha) para prevenir ataques de enumeração de usuários.
                logger.warning(f"Failed login attempt for email: {email} - Invalid credentials.")
                return jsonify({'message': 'Invalid email or password'}), 401
        
        except OperationalError as e:
            # Erro na conexão ou operação do banco de dados.
            logger.error(f"Database operational error during login for {email}: {e}", exc_info=True)
            return jsonify({'message': 'Database error during login. Please try again later.'}), 500
        except SQLAlchemyError as e:
            # Captura outros erros específicos do SQLAlchemy.
            logger.error(f"A SQLAlchemy error occurred during login for {email}: {e}", exc_info=True)
            return jsonify({'message': 'A database error occurred. Please try again.'}), 500
        except Exception as e:
            # Captura outros erros inesperados para evitar que a aplicação quebre.
            logger.error(f"An unexpected error occurred during login for {email}: {e}", exc_info=True)
            return jsonify({'message': 'An unexpected error occurred. Please try again.'}), 500
    else:
        # A validação do formulário falhou (ex: email inválido, senha muito curta, CSRF inválido).
        logger.warning(f"Form validation failed for login. Errors: {form.errors}")
        # Retorna os erros de validação para o cliente.
        return jsonify({'message': 'Invalid request data', 'errors': form.errors}), 400

@auth.route('/register', methods=['POST'])
# Assim como no login, se o formulário (FlaskForm) é usado, `form.validate_on_submit()` já faz a validação CSRF.
def register():
    """
    Endpoint para registro de novo usuário.
    Recebe email e senha, hashea a senha e salva o novo usuário no banco de dados.
    """
    data = request.get_json() if request.is_json else request.form
    form = RegistrationForm(data=data)

    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        try:
            # A validação de email único já deve ser feita no formulário WTForms (`validate_email` no `RegistrationForm`).
            # No entanto, uma verificação adicional aqui (ou um `try-except IntegrityError`) é uma boa prática
            # para lidar com condições de corrida ou falhas na validação do formulário.
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                logger.warning(f"Registration attempt with existing email: {email}")
                return jsonify({'message': 'Email already registered'}), 409 # Conflict
            
            # Hashing da senha usando bcrypt.gensalt() para gerar um salt único e seguro.
            # O salt é incorporado ao hash resultante, tornando cada hash único, mesmo para senhas idênticas.
            # O método `set_password` no modelo `User` encapsula essa lógica.
            new_user = User(email=email)
            new_user.set_password(password) # Usa o método do modelo para hash da senha
            
            db.session.add(new_user)
            db.session.commit() # Tenta commitar a transação no banco de dados.
            
            logger.info(f"User {email} registered successfully with ID: {new_user.id}")
            return jsonify({'message': 'User created successfully', 'user_id': new_user.id, 'email': new_user.email}), 201 # Created
        
        except IntegrityError as e:
            # Catcha erros de integridade de banco de dados (ex: violação de unique constraint para email).
            # Isso pode ocorrer em condições de corrida se dois usuários tentarem registrar o mesmo email
            # simultaneamente, mesmo com a validação `validate_email` no formulário.
            db.session.rollback() # Garante que a transação é desfeita em caso de erro.
            logger.error(f"Integrity error during user registration for {email}: {e}", exc_info=True)
            # Mensagem genérica para evitar vazar detalhes do banco de dados.
            return jsonify({'message': 'Registration failed due to a data integrity issue (e.g., email already in use).'}), 409 # Conflict
        except OperationalError as e:
            # Erro de conexão ou operação com o banco de dados.
            db.session.rollback()
            logger.error(f"Database operational error during registration for {email}: {e}", exc_info=True)
            return jsonify({'message': 'Database error during registration. Please try again later.'}), 500
        except SQLAlchemyError as e:
            # Captura outros erros específicos do SQLAlchemy.
            db.session.rollback()
            logger.error(f"A SQLAlchemy error occurred during registration for {email}: {e}", exc_info=True)
            return jsonify({'message': 'A database error occurred during registration. Please try again.'}), 500
        except Exception as e:
            # Outros erros inesperados durante o registro.
            db.session.rollback() # Sempre rollback em caso de erro para não deixar transações pendentes.
            logger.error(f"An unexpected error occurred during registration for {email}: {e}", exc_info=True)
            return jsonify({'message': 'An unexpected error occurred. Please try again.'}), 500
    else:
        # Validação do formulário falhou (ex: email inválido, senha muito curta, senhas não coincidem).
        logger.warning(f"Form validation failed for registration. Errors: {form.errors}")
        return jsonify({'message': 'Invalid request data', 'errors': form.errors}), 400

@auth.route('/logout')
@login_required # Garante que apenas usuários logados podem acessar este endpoint.
def logout():
    """
    Endpoint para logout de usuário.
    Invalida a sessão do usuário logado.
    """
    if current_user.is_authenticated:
        user_email = current_user.email
        logout_user() # Função do Flask-Login para deslogar o usuário, limpando a sessão.
        logger.info(f"User {user_email} logged out successfully.")
        # Para aplicações web, você redirecionaria para a página inicial e mostraria uma mensagem flash.
        return jsonify({'message': 'Logged out successfully'}), 200
    else:
        # Se por algum motivo (ex: cache de browser, sessão expirada) um usuário não autenticado tentar fazer logout.
        logger.warning("Attempt to logout by an unauthenticated user or session already expired.")
        return jsonify({'message': 'Not logged in or session expired'}), 401

@auth.route('/status', methods=['GET'])
@login_required # Exemplo de endpoint que requer autenticação para ser acessado.
def status():
    """
    Exemplo de endpoint protegido que retorna o status de login do usuário atual.
    Demonstra o uso de `current_user` para acessar informações do usuário logado.
    """
    logger.info(f"User {current_user.email} accessed status endpoint.")
    return jsonify({
        'message': 'You are logged in!',
        'user_id': current_user.id,
        'email': current_user.email,
        'is_authenticated': current_user.is_authenticated,
        'is_fresh': current_user.is_fresh # Indica se o login é "fresh" (recente)
    }), 200

@auth.route('/protected_fresh_login_required')
@fresh_login_required
def protected_fresh_login_required():
    """
    Exemplo de endpoint que requer um login "fresh".
    Isso significa que o usuário deve ter feito login recentemente (ou re-autenticado)
    e não está apenas usando um cookie de "remember me".
    Útil para operações sensíveis como mudança de senha ou configurações de segurança.
    """
    logger.info(f"User {current_user.email} accessed fresh login required endpoint.")
    return jsonify({
        'message': 'This endpoint requires a fresh login!',
        'user_id': current_user.id,
        'email': current_user.email
    }), 200


# --- Funções de Inicialização ---

def init_login_manager(app):
    """
    Inicializa o Flask-Login com a aplicação Flask.
    Esta função deve ser chamada na inicialização da sua aplicação principal.
    """
    login_manager.init_app(app)
    logger.info("Flask-Login initialized.")

def init_auth(app):
    """
    Inicializa o Blueprint de autenticação e o CSRFProtect com a aplicação Flask.
    Esta é a função principal para integrar este módulo na sua aplicação Flask.
    """
    app.register_blueprint(auth)
    # CSRFProtect precisa do objeto 'app' para sua inicialização completa.
    # Isso configura o tratamento de tokens CSRF para toda a aplicação.
    csrf.init_app(app) 
    init_login_manager(app) # Garante que o login manager também seja inicializado.
    logger.info("Authentication blueprint and CSRFProtect initialized.")

