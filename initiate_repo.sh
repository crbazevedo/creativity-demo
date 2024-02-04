#!/bin/bash

# Initialize a new git repository
git init

# Add all files in the current directory to the repository
git add .

# Commit the changes with an initial commit message
git commit -m "Repo folder structure and stubs"

# Create a new remote repository on a hosting service like GitHub or GitLab

# Add the URL of the remote repository as the origin
git remote add origin git@github.com:crbazevedo/creativity-demo.git

# Push the local repository to the remote repository
git push -u origin master
