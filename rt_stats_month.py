import numpy as np


def calculate_return_over_period(returns_array_reverse, months):
    if len(returns_array_reverse) > months:
        return returns_array_reverse[months]
    return np.nan


def calculate_omega_ratio(returns_array, threshold=0):
    excess_returns = returns_array - threshold
    positive_returns = excess_returns[excess_returns > 0].sum()
    negative_returns = -excess_returns[excess_returns < 0].sum()

    if negative_returns == 0:
        return np.nan if positive_returns == 0 else np.inf

    return positive_returns / negative_returns


def calculate_var(returns_array, confidence_level=0.95):
    return np.percentile(returns_array, (1 - confidence_level) * 100)


def compute_cum_max(arr):
    cum_max = np.empty_like(arr)
    current_max = arr[0]
    for i in range(len(arr)):
        if arr[i] > current_max:
            current_max = arr[i]
        cum_max[i] = current_max
    return cum_max


def rolling_window_mean(arr, window):
    result = np.empty(len(arr) - window + 1)
    for i in range(len(result)):
        result[i] = arr[i:i + window].mean()
    return result


def rolling_window_std(arr, window):
    result = np.empty(len(arr) - window + 1)
    for i in range(len(result)):
        result[i] = arr[i:i + window].std()
    return result


def metrics(returns_array: np.ndarray, full=False):
    total_months = len(returns_array)

    # 최근 기간 수익률 처리
    returns_array_reverse = np.cumprod(1 + returns_array[::-1]) - 1

    # 누적 수익률
    cumulative_return = returns_array_reverse[-1]

    # 특정 기간 수익률 계산 (월 기준)
    periods = [1, 3, 6, 12, 24]  # 월 단위로 1개월, 3개월, 6개월, 12개월, 24개월
    returns_over_periods = {f'rt_{m}m': calculate_return_over_period(returns_array_reverse, m) for m in periods}

    # CAGR (연평균 수익률)
    years = total_months / 12  # 월 단위에서 연 단위로 변환
    cagr = (cumulative_return + 1) ** (1 / years) - 1

    # MDD (최대 낙폭)
    cum_returns = np.cumprod(1 + returns_array)
    peak = compute_cum_max(cum_returns)
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min()

    # 평균 수익률
    mean_return = returns_array.mean()

    # 변동성 (월 단위 변동성을 연 단위 변동성으로 변환)
    volatility = returns_array.std() * np.sqrt(12)

    # 샤프 비율 (무위험 수익률은 0으로 가정)
    sharpe_ratio = mean_return / volatility * np.sqrt(12)

    # 승리한 달과 패배한 달의 수 계산
    win_months = np.sum(returns_array > 0)
    loss_months = np.sum(returns_array < 0)

    # 수익 평균 수익률과 손실 평균 수익률 계산
    mean_win_return = np.mean(returns_array[returns_array > 0])
    mean_loss_return = np.mean(returns_array[returns_array < 0])

    # 기본적인 통계량 딕셔너리 생성
    stats = {
        'cumulative_return': cumulative_return,
        'cagr': cagr,
        'mdd': mdd,
        'mean': mean_return,
        'volatility': volatility,
        'sharpe': sharpe_ratio,
        'mean_win_return': mean_win_return,
        'mean_loss_return': mean_loss_return,
        'total_months': total_months,
        'win_months': win_months,
        'loss_months': loss_months,
    }

    # full 옵션이 True일 때만 추가 통계량을 계산
    if full:
        # 3개월 롤링 수익률의 평균 및 표준편차 계산
        rolling_mean_3m = rolling_window_mean(returns_array, 3).mean()
        rolling_std_3m = rolling_window_std(returns_array, 3).mean()

        # Recovery Factor
        recovery_factor = cumulative_return / abs(mdd) if mdd != 0 else np.nan

        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        # Serenity Index
        serenity_index = cagr / (ulcer_index * volatility) if ulcer_index != 0 else np.nan

        # Longest DD Months
        drawdown_duration = 0
        max_drawdown_duration = 0
        for val in drawdown:
            if val < 0:
                drawdown_duration += 1
                if drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = drawdown_duration
            else:
                drawdown_duration = 0
        longest_dd_months = max_drawdown_duration

        # Calmar Ratio
        calmar_ratio = cagr / abs(mdd) if mdd != 0 else np.nan

        # Skewness
        skewness = ((returns_array - mean_return) ** 3).mean() / (returns_array.std() ** 3)

        # Kurtosis
        kurtosis = ((returns_array - mean_return) ** 4).mean() / (returns_array.std() ** 4) - 3

        # Omega Ratio
        omega_ratio = calculate_omega_ratio(returns_array)

        # Monthly VaR
        monthly_var = calculate_var(returns_array)

        # 추가 통계량을 stats 딕셔너리에 추가
        stats.update({
            'ulcer_index': ulcer_index,
            'monthly_var': monthly_var,
            'skew': skewness,
            'kurtosis': kurtosis,
            'longest_dd_months': longest_dd_months,
            'recovery_factor': recovery_factor,
            'serenity_index': serenity_index,
            'calmar': calmar_ratio,
            'omega': omega_ratio,
            'rolling_mean_3m': rolling_mean_3m,
            'rolling_std_3m': rolling_std_3m,
            'longest_dd_months_ratio': longest_dd_months / total_months,
            'win_months_ratio': win_months / total_months,
            'loss_months_ratio': loss_months / total_months,
        })

    # 특정 기간 수익률 추가
    stats.update(returns_over_periods)

    return stats