from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class CustomUserCreationForm(UserCreationForm):
    username = forms.CharField(
        label="用户名",
        strip=False,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        help_text=""
    )
    password1 = forms.CharField(
        label="密码",
        strip=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text=""
    )
    password2 = forms.CharField(
        label="确认密码",
        strip=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text=""
    )

    class Meta:
        model = User
        fields = ['username',  'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super(CustomUserCreationForm, self).__init__(*args, **kwargs)

        self.fields['username'].widget.attrs.update({'class': 'form-control'})
