from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("SignupAction", views.SignupAction, name="SignupAction"),
               path("Signup.html", views.Signup, name="Signup"),
               path("TrainML", views.TrainML, name="TrainML"),
	       path("ViewUser", views.ViewUser, name="ViewUser"),
	       path("ViewChats", views.ViewChats, name="ViewChats"),
	       path("TextChatbot.html", views.TextChatbot, name="TextChatbot"),
	       path("ChatData", views.ChatData, name="ChatData"),
	       path("ApplyLoanAction", views.ApplyLoanAction, name="ApplyLoanAction"),
               path("ApplyLoan.html", views.ApplyLoan, name="ApplyLoan"),
	       path("ViewApplications", views.ViewApplications, name="ViewApplications"),
	       path("UpdateStatus", views.UpdateStatus, name="UpdateStatus"),
	       path("ViewStatus", views.ViewStatus, name="ViewStatus"),
]
