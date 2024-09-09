import asyncio
from datetime import datetime, timedelta
from nio import AsyncClient, LoginResponse, RoomMessageText, RoomMessageNotice

async def login(username, password, homeserver):
    client = AsyncClient(homeserver, username)
    response = await client.login(password)
    if isinstance(response, LoginResponse):
        print("Logged in!")
        return client
    else:
        print("Failed to log in:", response)
        return None

async def get_messages(client, room_id, since_date):
    # Initial request to get the latest end token
    end_token = (await client.room_messages(room_id, start="", limit=1)).end
    
    messages = []
    while True:
        chunk = await client.room_messages(room_id, start=end_token, limit=100)
        if not chunk.chunk:
            break
        for event in chunk.chunk:
            timestamp = datetime.utcfromtimestamp(event.server_timestamp / 1000)
            if timestamp < since_date:
                return messages
            if isinstance(event, (RoomMessageText, RoomMessageNotice)):
                messages.append({
                    "sender": event.sender,
                    "body": event.body,
                    "timestamp": timestamp.isoformat()
                })
        end_token = chunk.end
    return messages

async def main(username, password, homeserver, room_id, since_date_str):
    since_date = datetime.fromisoformat(since_date_str)
    client = await login(username, password, homeserver)
    if client:
        messages = await get_messages(client, room_id, since_date)
        await client.close()
        return messages
    return []

def messages_to_multiline_string(messages):
    lines = []
    for message in messages:
        line = f"Sender: {message['sender']}\nMessage: {message['body']}\n"
        lines.append(line)
    return "\n".join(lines)

if __name__ == "__main__":
    username = "@dream_bot_1:matrix.org"
    password_file_path = '/data/common/configs/bot/dream_bot_psw'
    with open(password_file_path, 'r') as file:
        bot_password = file.read()
    password = bot_password
    homeserver = "https://matrix.org"
    room_id = "!vPhQLHJayUphCPylyu:matrix.org" #!kgqOAITmdRxyeisaMp:matrix.org
    # Calculate since_date_str to be 24 hours into the past
    hours_into_the_past = 24 * 14
    since_date = datetime.utcnow() - timedelta(hours=hours_into_the_past)
    since_date_str = since_date.isoformat()

    messages = asyncio.run(main(username, password, homeserver, room_id, since_date_str))
    messages[0]
    messages_str = messages_to_multiline_string(messages)


    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "The context of the period to consider is:" + messages_str},
        {"role": "user", "content": "Sumarise this context in a poem, ensuring all senders are considered and feature. be as bizare as possible"}
    ]
    )

    print(completion.choices[0].message.content)

    # total_tokens = completion.usage.total_tokens
    # print(f"\nTotal number of tokens used: {total_tokens}")
