echo "Configuring git to treat working directory as safe..."
git config --global --add safe.directory $(pwd)

if [ $? -eq 0 ]; then
    echo "Successfully configured git!"
else
    echo "Failed to configure git to treat working directory as safe."
fi

echo ""
echo "Adding poetry to path..."
poetry_path="/home/developer/.local/bin"
if [ -d "$poetry_path" ] && [[ ":$PATH:" != *":$poetry_path:"* ]]; then
    PATH="${PATH:+"$PATH:"}$poetry_path"
fi

echo "Checking poetry version..."
echo "Using poetry version:"
poetry --version

if [ $? -eq 0 ]; then
    echo "Successfully installed poetry!"
else
    echo "Failed to install poetry."
fi

echo ""
echo "Creating alias localpy to local Python environment..."

echo 'alias localpy="$(pwd)/.venv/bin/python"' >> ~/.bashrc

if [ $? -eq 0 ]; then
    echo "Alias created! To run a script with your local python use:"
    echo "localpy path/to/script.py"
else
    echo "Failed to create alias."
fi

echo ""
echo "Logged in as user:"
whoami

echo ""
echo "Success - ready to develop!"
echo ""