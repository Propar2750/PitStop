import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.competition_download('playground-series-s6e5')

print("Path to competition files:", path)