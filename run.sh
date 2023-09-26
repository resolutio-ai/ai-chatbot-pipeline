while [ true ]
do
    curl -X POST 'http://127.0.0.1:5000/bot' -H "Content-Type: application/json"
    if [[ "$?" -eq 0 ]]; then
        exit 0
    fi
done
