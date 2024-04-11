import os
import anthropic

def main(save_path):
    with open('system_prompt_get_prompts.txt', 'r') as file:
        anthropic_system_prompt = file.read()

    with open('user_prompt_get_prompts.txt', 'r') as file:
        anthropic_user_prompt = file.read()

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=3200,
        system=anthropic_system_prompt,
        messages=[
            {"role": "user", "content": anthropic_user_prompt}
        ]
    )

    with open(f'{save_path}prompts.txt', 'w') as file:
        file.write(message.content[0].text)

if __name__ == "__main__":
    main()