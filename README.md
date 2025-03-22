while true; do
    git add model_data.pkl
    git commit -m "Auto-update model_data.pkl at $(date)"
    git push origin main
    sleep 10
done
