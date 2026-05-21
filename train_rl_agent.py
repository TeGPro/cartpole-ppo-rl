import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

print("Создаем симулятор балансировки шеста")
env = gym.make("CartPole-v1")

model = PPO(
    policy="MlpPolicy", 
    env=env, 
    learning_rate=0.001, 
    verbose=1, 
    device="auto"
)

print("🏋️‍♂️ Начинаем обучение агента...")
model.learn(total_timesteps=10000)

# Сохраняем веса
model.save("ppo_cartpole")
print("Модель сохранена в файл 'ppo_cartpole.zip'")
env.close()

print("\nЗапускаем проверку обученного ИИ...")
env_render = gym.make("CartPole-v1", render_mode="human")

# Загружаем модель обратно
trained_model = PPO.load("ppo_cartpole", env=env_render)

# Оцениваем
mean_reward, std_reward = evaluate_policy(trained_model, env_render, n_eval_episodes=3)
print(f"Средний результат ИИ: {mean_reward:.2f} из 500.00 очков!")

# Показываем один раунд вживую
obs, info = env_render.reset()
done = False

while not done:
    action, _states = trained_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_render.step(action)
    done = terminated or truncated

env_render.close()
print("Тестирование завершено!")
