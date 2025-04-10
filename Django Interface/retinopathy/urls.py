from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    # path('analyzecsv', views.analyzecsv, name='analyzecsv')
    # path('upload', views.upload_audio, name='upload_audio')
    # path('signup',views.userSignup,name = 'userSignup'),
    # path('login',views.userLogin,name = 'userLogin'),
    # path('logout',views.userLogout,name = 'userLogout'),
    # path('analyzer',views.analyzer,name = 'analyzer'),
    # path('generate',views.getText,name = 'getText'),
    # path('builder',views.builder,name = 'builder'),
    # path('builderform',views.builderform,name = 'builderform'),
    # path('resume',views.test,name = 'resume'),
    # path('aboutus',views.aboutus,name = 'aboutus'),
    # path('signin',views.signin,name = 'signin'),
]