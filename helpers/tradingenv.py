import gym
import numpy as np
import math
import torch
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, price_data, tickers, num_features=0, sequence_len=0, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.price_data = price_data
        self.tickers = tickers  # This should be defined before usage or passed as a parameter
        self.lambda_penalty = 0.002
        self.initial_balance = initial_balance
        self.current_step = 0  # Start at 201 for history
        # self.max_steps = len(self.data[self.data["TICKER"] == my_stocks[0]]) - self.current_step
        self.max_steps = len(self.data) - self.current_step
        self.cash_balance = self.initial_balance
        self.positions = np.zeros(len(self.tickers))
        self.last_prices = np.zeros(len(self.tickers))
        self.total_reward = 1.0
        self.average_buy_prices = np.zeros(len(self.tickers))
        self.buy_signal_counter = 0
        self.sell_signal_counter = 0
        
        self.comission = 0.004
       
        # For reward normalization
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.alpha = 0.01  # smoothing factor
        
        self.use_prev_action = True
        self.use_comission = True
        self.punish_negatives_extra = False
        self.include_singular_rewards = False

        self.compare_reward_to_mean = False
        self.add_punishment_for_deviating_from_mean = False
        self.add_punishment_for_deviating_from_mean_divider = 1000

        self.signals_to_accummulate = 1

        self.single_stock_reward = True
        self.normalize_reward = False
        self.daily_reward = True
        self.single_stock_portfolio_change = 1.0
        self.hold_multiplier = 0.0
        self.sell_on_hold = 0.0
        
        self.buy_only_for_whats_sold = False

        self.use_only_buy_sell = True
        self.divide_by_volatility = True
        self.use_both_dense_and_cnn = False

        if self.single_stock_reward:
            self.punish_negatives_extra = False
            self.use_prev_action = False
            self.normalize_reward = True
            self.look_after_days_for_reward = 40
            self.gamma = 0.8

        if self.divide_by_volatility:
            self.upper_cap = 1.2
            self.lower_cap = 0.8
        else:
            self.upper_cap = 1.2
            self.lower_cap = 0.8
            
        self.cash_interest = 0.04

        action_size = len(self.tickers)
        
        if not self.single_stock_reward:
            action_size += 1
        
        enlarged_action_size = action_size
        if self.include_singular_rewards:
            enlarged_action_size += len(self.tickers)
            low = np.array([0.0] * action_size + [-0.05] * len(self.tickers), dtype=np.float32)
            high = np.array([0.0] * action_size + [0.05] * len(self.tickers), dtype=np.float32)
        else:
            # Number of discrete bins
            bins = 3  # gives 0.0, 0.1, ..., 1.0
            if self.use_only_buy_sell:
                bins = 2
            
            self.action_space = spaces.Discrete(bins)
            
        
        print(f"self.action_space.shape: {self.action_space.shape}")
        self.prev_action = [1.0 , 0.0]
        
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.tickers), sequence_len, num_features), dtype=np.float32),
            "prev_action": spaces.Box(low=0.0, high=1.0, shape=(len(self.tickers) + 1,), dtype=np.float32)
        })

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        
        action_size = len(self.tickers) + 1
        
        self.current_step = 0
        self.tickers = self.tickers
        self.cash_balance = self.initial_balance
        self.positions = np.zeros(len(self.tickers))
        self.last_prices = np.zeros(len(self.tickers))
        self.total_reward = 1.0
        self.average_buy_prices = np.zeros(len(self.tickers))
        self.prev_action = [1.0 , 0.0]
        self.buy_signal_counter = 0
        self.sell_signal_counter = 0
        
        return self._next_observation()

    def _next_observation(self):
        
        return {
            "obs": self.data[self.current_step],  # Current observation
            "prev_action": self.prev_action  # Previously taken action
        }
    def total_balance(self):
        return np.sum(self.positions * self.last_prices) + self.cash_balance
        
    def calculate_reward(self, action, cash_balance, current_step, positions, last_prices, average_buy_prices, price_data, tickers, disable_comission=False):
    
        print(f"calculate_reward: action: {action}")
        print(f"calculate_reward: cash_balance: {cash_balance}")
        print(f"calculate_reward: current_step: {current_step}")
        print(f"calculate_reward: positions: {positions}")
        print(f"calculate_reward: last_prices: {last_prices}")
    
    
        positions = positions.copy()
        last_prices = last_prices.copy()
        average_buy_prices = average_buy_prices.copy()
        
        penalty_strength = 0.03 / 2   # Tune this value to adjust punishment effect
        ideal_value = 1 / 2  # Added missing variable
        penalty = 0  # -penalty_strength * np.sum(np.abs(action - ideal_value))

        today_value = cash_balance
        tomorrow_value_if_no_change = cash_balance + (cash_balance * 0.04 / 365)
        tomorrow_value_with_change = 0
        tomorrow_value_with_change_capped = 0
        reward = 0

        if self.use_comission and not disable_comission:
            comission = self.comission
        else:
            comission = 0.0

        for i, ticker in enumerate(self.tickers):
            today_stock_price = price_data[i, current_step]
            tomorrow_stock_price = price_data[i, current_step + 1]
            tomorrow_stock_price_capped = max(min(tomorrow_stock_price, today_stock_price * self.upper_cap), today_stock_price * self.lower_cap)
            today_value += today_stock_price * positions[i]
            tomorrow_value_if_no_change += tomorrow_stock_price_capped * positions[i]
            # print(f"current step: {self.current_step} ticker: {ticker}, today_stock_price: {today_stock_price}")

        # print(action)
        for i, act in enumerate(action):
            if i == len(action) - 1:
                ## MEANS THE CASH TO HOLD
                break
                
            ticker = tickers[i]
            
            today_stock_price = price_data[i, current_step]
            tomorrow_stock_price = price_data[i, current_step + 1]
            tomorrow_stock_price_capped = max(min(tomorrow_stock_price, today_stock_price * self.upper_cap), today_stock_price * self.lower_cap)
            
            today_stock_allocation = (today_stock_price * positions[i]) / today_value
            last_prices[i] = today_stock_price
            
            if act > today_stock_allocation and act > 0:
                desired_invest = act * today_value
                current_invest = today_stock_price * positions[i]
                diff = desired_invest - current_invest
                amount_to_buy = math.floor((diff / today_stock_price) * (1 - comission))
                if amount_to_buy > 0:
                    positions[i] += amount_to_buy
                    cash_balance -= (today_stock_price * amount_to_buy) * (1 + comission)
                    if positions[i] > 0:
                        average_buy_prices[i] = ((average_buy_prices[i] * (positions[i] - amount_to_buy)) + (today_stock_price * amount_to_buy)) / positions[i]  # Fixed index i-1 to i
                    else:
                        average_buy_prices[i] = today_stock_price  # Changed current_price to today_stock_price
            elif act < today_stock_allocation:
                desired_invest = act * today_value
                current_invest = today_stock_price * positions[i]
                diff = current_invest - desired_invest
                amount_to_sell = math.ceil(diff / today_stock_price)
                if amount_to_sell > 0:
                    positions[i] -= amount_to_sell
                    cash_balance += (today_stock_price * amount_to_sell) * (1 - comission)
                    
            tomorrow_value_with_change += tomorrow_stock_price * positions[i]
            tomorrow_value_with_change_capped += tomorrow_stock_price_capped * positions[i]

        
        cash_balance = cash_balance + cash_balance * self.cash_interest / 365
        tomorrow_value_with_change += cash_balance
        tomorrow_value_with_change_capped += cash_balance

        reward += tomorrow_value_with_change_capped / tomorrow_value_if_no_change - 1.0
        reward += penalty
        
        if self.punish_negatives_extra and reward < 0:
            reward *= 2
        return reward, cash_balance, positions, last_prices, average_buy_prices

    def calculate_daily_loss(self, actions, current_step, price_data, tickers):
        loss = 0
        # actions = actions * 0.2 - 0.1
        for i, ticker in enumerate(self.tickers):
            today_stock_price = price_data[i, current_step]
            tomorrow_stock_price = price_data[i, current_step + 1]
            diff = (tomorrow_stock_price / today_stock_price - 1.0)
            
            print(f"Loss for ticker: {ticker}: {actions[i] - diff} .. action: {actions[i]} .. diff: {diff} at current step: {current_step}")
            loss += abs(actions[i] - diff)
            
        # print(f"Loss for {loss} at current step: {current_step}")
        loss = loss / len(actions)
        loss = loss * loss / 100
        return loss

    def normalize_reward(self, reward):
        return reward
        # np.sign(reward) * (abs(reward) ** 0.4)
        # return reward + 1
    
    def calculate_reward_for_day(self, current_step, day):
        tomorrow_stock_price = self.price_data[0, self.current_step + day + 1]
        current_stock_price = self.price_data[0, self.current_step + day]
        tomorrow_price_capped = max(min(tomorrow_stock_price, current_stock_price * self.upper_cap), current_stock_price * self.lower_cap)
        
        ratio_capped = ((tomorrow_price_capped / current_stock_price) - 1.0) * (self.gamma ** day)
        
        if self.divide_by_volatility:
            volatility = self.data[self.current_step + day][-1][-1, 10] + 1e-8
            print(f"volatility: {volatility}")
        else:
            volatility = 1.0

        return ratio_capped / volatility
        
    def step(self, action):
        orig_action = action

        if self.signals_to_accummulate > 1:
            if action == 0:
                self.sell_signal_counter += 1
                self.buy_signal_counter = 0
                if self.sell_signal_counter >= self.signals_to_accummulate:
                    action = 0
                else:
                    action = 1
            elif action == 2:
                self.sell_signal_counter = 0
                self.buy_signal_counter += 1
                if self.buy_signal_counter >= self.signals_to_accummulate:
                    action = 2
                else:
                    action = 1
            else:
                self.sell_signal_counter = 0
                self.buy_signal_counter = 0

        # print(f"action: {action}")
        if self.include_singular_rewards:
            singular_actions = action[-len(self.tickers):]
            action = action[:len(self.tickers)+1]

        if self.single_stock_reward:
            if self.use_only_buy_sell:
                if action == 0:
                    action = -1
                else:
                    action = 1
                final_state = max(0.0, min(1.0, self.prev_action[0] + (action * self.single_stock_portfolio_change)))
                action = [final_state, 1.0 - final_state]
            else:
                # print(f"action: {action}")
                action = action - 1
                if self.sell_on_hold != 0.0 and action == 0:
                    action = self.sell_on_hold
                final_state = max(0.0, min(1.0, self.prev_action[0] + (action * self.single_stock_portfolio_change)))
                action = [final_state, 1.0 - final_state]
                # print(f"post action: {action}")

        if not self.buy_only_for_whats_sold:
            # action = action * 0.2 + 0.05
            # print(f"actions: {action}")
            epsilon = 1e-8  # Small constant to prevent division by zero
    #        action = np.array(action)  # Ensure action is a numpy array
            action = action.astype(float)  # Convert to float type first
            action /= (np.sum(action) + epsilon)
        
            ideal_value = 1 / (len(self.tickers))  # 0.0909 for 11 actions
            ideal_actions = ([ideal_value] * len(self.tickers)) + [0.0]

            
            # print(f"action: {action}")
            default_reward, _ , _ , _ , _ = self.calculate_reward(ideal_actions, self.cash_balance, self.current_step, self.positions.copy(), self.last_prices.copy(), self.average_buy_prices.copy(), self.price_data, self.tickers, disable_comission=True)
            
            my_reward, cash_balance, positions, last_prices, average_buy_prices = self.calculate_reward(action, self.cash_balance, self.current_step, self.positions, self.last_prices, self.average_buy_prices, self.price_data, self.tickers)
            
            self.cash_balance = cash_balance
            self.positions = positions
            self.last_prices = last_prices
            self.average_buy_prices = average_buy_prices
        else:
            my_reward = 0
            number_of_buys = 0
            for i, act in enumerate(action):
                if i == len(action) - 1:
                    ## MEANS THE CASH TO HOLD
                    break
                today_stock_price = self.price_data[i, self.current_step]
                self.last_prices[i] = today_stock_price
                if act == 0:
                    self.cash_balance += (today_stock_price * self.positions[i]) * (1 - self.comission)
                    self.positions[i] = 0
                else:
                    number_of_buys += 1
                    
            if number_of_buys > 0:
                n = len(self.tickers)
                valid_indices = [i for i in range(n) if action[i] > 0 and self.positions[i] * self.last_prices[i] / self.total_balance() > 3 / len(self.tickers)]
                if valid_indices:
                    for i in range(n):
                        if i in valid_indices and action[i] > 0:
                            print(f"OVERRIDINGNN>>>>")
                            action[i] = 0

            
                buy_per_stock = self.cash_balance / number_of_buys
                buy_per_stock = min(buy_per_stock, 3 * self.total_balance() / (len(self.tickers)))
                n = len(self.positions)
                
                for i, act in enumerate(action):
                    if i == len(action) - 1:
                        ## MEANS THE CASH TO HOLD
                        break
                    if act > 0:
                        today_stock_price = self.price_data[i, self.current_step]
#                        buy_per_stock = min(buy_per_stock, 3 * self.total_balance() / (len(self.tickers)) - self.positions[i] * self.last_prices[i])
                        amount_to_buy = math.floor((buy_per_stock / today_stock_price) * (1 - self.comission))
                        self.cash_balance -= (today_stock_price * amount_to_buy) * (1 + self.comission)
                        self.positions[i] += amount_to_buy

        # reward = 1000 * (torch.log(torch.tensor((my_reward - default_reward) + 1)) - loss)
        if self.compare_reward_to_mean:
            reward = my_reward - default_reward
            reward = 1000 * torch.log(torch.tensor(reward + 1))
        elif self.include_singular_rewards:
            loss = self.calculate_daily_loss(singular_actions, self.current_step, self.price_data, self.tickers)
            reward = my_reward - default_reward - loss
            reward *= 1000
        elif self.add_punishment_for_deviating_from_mean:
            diff_sum = np.sum(np.abs(action - ideal_actions))
            reward = my_reward - (diff_sum / self.add_punishment_for_deviating_from_mean_divider)
            reward = 1000 * torch.log(torch.tensor(reward + 1))
        else:
            reward = my_reward
            # print(f"my_reward: {my_reward}")
            reward = 1000 * torch.log(torch.tensor(reward + 1))


        if self.single_stock_reward and self.daily_reward:
            
            if orig_action == 0:
                orig_action = -1.0
            elif orig_action == 1:
                if self.use_only_buy_sell:
                    orig_action = 0.0
                else:
                    orig_action = 0.0
            else:
                orig_action = 1.0

            reward = 0.0
            for i in range(self.look_after_days_for_reward):
                reward += (orig_action * self.calculate_reward_for_day(self.current_step, i))

            reward = 1000 * torch.log(torch.tensor(reward + 1e-8))  # Added small epsilon to avoid log(0)
            # reward -= 1.0
            self.current_step += 1
            # print(f"self.data.shape[0]: {self.data.shape[0]}")
            done = self.current_step >= self.data.shape[0] - self.look_after_days_for_reward
        else:
            self.current_step += 1
            # print(f"self.data.shape[0]: {self.data.shape[0]}")
            done = self.current_step >= self.data.shape[0] - 8
        
        # Store action for next timestep
        self.prev_action = action

        # print(f"reward: {reward}")
        
        obs = self._next_observation()
        self.total_reward *= reward

        # print(f"reward: {reward}, total_reward: {self.total_reward}")
        return obs, reward, done, {}
