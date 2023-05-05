from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from .forms import LoginForm, RegisterForm


def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/stock')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


# def user_register(request):
#     if request.method == 'POST':
#         form = RegisterForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('login')
#     else:
#         form = RegisterForm()
#     return render(request, 'register.html', {'form': form})

from .forms import CustomUserCreationForm

def user_register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'login.html', {'form': form})
    # return redirect('/login')



def user_logout(request):
    logout(request)
    # form = CustomUserCreationForm()
    # form = LoginForm()
    # return render(request, 'login.html', {'form': form})

    return redirect('/login')
