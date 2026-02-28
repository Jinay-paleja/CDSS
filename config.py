import os

class Config:
    """Production configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'heartcare-ai-secret-key-2024'
    DEBUG = False
    TESTING = False

class DevelopmentConfig:
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig:
    """Testing configuration"""
    DEBUG = True
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': Config,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
