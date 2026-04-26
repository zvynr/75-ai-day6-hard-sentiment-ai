import os

class Config: 
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
    API_TITLE = '75 Hard AI Engineering Challenge API'
    ENGINEER = 'Annisa'
class DevelopmentConfig(Config): 
    DEBUG = True, 
    ENVIRONMENT = 'development'
class ProductionConfig(Config):
    DEBUG = False
    ENVIRONMENT = 'production'
config = {
    'development' : DevelopmentConfig, 
    'production' : ProductionConfig,
    'default': DevelopmentConfig
}