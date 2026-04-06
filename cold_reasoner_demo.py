"""
ColdReasoner + CEAL: 基于贝叶斯理性分析和合规规则库的 RAMTN 安全循环
建构者: 模拟 LLM (可配置谄媚率)
质疑者: CEAL 规则库 + 贝叶斯更新
观察者: 贝叶斯后验信念
"""

import numpy as np

# ==================== 世界模型参数 ====================
TRUE_H = 1                # 真实世界状态 H=1
K = 2                     # 每轮数据点数量
P_D1_GIVEN_H0 = 0.4
P_D1_GIVEN_H1 = 0.6
likelihood = np.array([[1-P_D1_GIVEN_H0, 1-P_D1_GIVEN_H1],
                       [P_D1_GIVEN_H0, P_D1_GIVEN_H1]])

def sample_data(h):
    """世界生成真实数据点 (长度为 K 的 0/1 数组)"""
    probs = [likelihood[1][h], likelihood[0][h]]
    return np.random.choice([1, 0], size=K, p=probs)

def impartial_response(data):
    """公正响应：随机选择一个真实数据点如实报告"""
    i = np.random.randint(0, K)
    return i, data[i]

def sycophantic_response(data, user_belief, allow_fabrication=True):
    """
    谄媚响应：选择最能验证用户当前信念的响应。
    user_belief: P(H=1)
    用户表达的观点 user_expr = 1 if user_belief > 0.5 else 0
    allow_fabrication: 是否允许虚构 (d 可以不等于 data[i])
    """
    user_expr = 1 if user_belief > 0.5 else 0
    best_rho = None
    best_post_expr = -1.0
    candidates = []
    if allow_fabrication:
        for i in range(K):
            for d in (0, 1):
                candidates.append((i, d))
    else:
        for i in range(K):
            candidates.append((i, data[i]))
    for i, d in candidates:
        L_H1 = likelihood[d][1] / K
        L_H0 = likelihood[d][0] / K
        prior = user_belief
        post_H1 = prior * L_H1 / (prior * L_H1 + (1 - prior) * L_H0)
        post_expr = post_H1 if user_expr == 1 else 1 - post_H1
        if post_expr > best_post_expr:
            best_post_expr = post_expr
            best_rho = (i, d)
    return best_rho

# ==================== 建构者 (模拟 LLM) ====================
def constructor_response(data, user_belief, sycophantic_prob=0.8):
    """
    模拟 LLM 响应。
    - 以 sycophantic_prob 概率输出谄媚响应（允许虚构）
    - 否则输出公正响应
    """
    if np.random.rand() < sycophantic_prob:
        return sycophantic_response(data, user_belief, allow_fabrication=True)
    else:
        return impartial_response(data)

# ==================== CEAL 规则库 ====================
def ceal_check(response, data):
    """
    CEAL 合规检查。
    返回 (是否合规, 违规原因)
    当前规则：
    1. 禁止虚构 (d != data[i])
    2. 禁止事实性谄媚 (即响应是谄媚策略产生的，这里通过检查响应是否等于公正响应来近似，
       但公正响应本身是随机的，无法直接比较。为了演示，我们直接使用一个更强的规则：
       如果响应不是公正响应，则视为违规。实际工程中可用符号模式匹配。）
       为简单起见，我们定义：任何非公正响应（即由 sycophantic_response 产生的）都视为违规。
       但这里无法直接知道响应来源，因此采用一个实用方法：如果响应与 impartial_response(data) 的结果相同，
       则认为是公正的；否则视为违规。注意：impartial_response 有随机性，需要固定随机种子对比不现实。
       更合理的模拟：在 CEAL 层级，我们直接禁止任何形式的谄媚，因此我们直接返回 False 除非响应是 impartial。
       由于我们无法准确判断，为了演示 CEAL 的效果，我们采用简化：仅拦截虚构。
       若要拦截事实性谄媚，需要更复杂的语义分析。这里我们扩展：如果响应是通过选择支持用户信念的数据点产生的，
       并且该数据点是真实的，那么它也属于违规。在模拟中，我们可以通过检查该响应是否等于
       sycophantic_response(data, user_belief, allow_fabrication=False) 来判断。
       但需要传入 user_belief，这里没有。因此我们仅实现虚构拦截。
       为了展示完整能力，我们实现一个更强的检查：如果响应中报告的值 d 不等于 data[i]（虚构），则拦截；
       另外，如果响应虽然不是虚构，但它是通过事实性谄媚产生的（即它刻意支持用户信念），我们也可以拦截，
       但需要额外的信息。这里我们只做虚构拦截，实际 CEAL 会包含更多规则。
    """
    i, d = response
    # 规则1: 禁止虚构
    if d != data[i]:
        return False, "fabrication"
    # 可以继续添加其他规则，例如主体性主张等，当前略
    return True, "compliant"

# ==================== 质疑者+观察者 (贝叶斯更新) ====================
def bayesian_observer(prior, response):
    """贝叶斯更新，返回后验信念和可信度（信息增量）"""
    i, d = response
    L_H1 = likelihood[d][1] / K
    L_H0 = likelihood[d][0] / K
    post_num = prior * L_H1
    post_den = post_num + (1 - prior) * L_H0
    if post_den == 0:
        posterior = prior
    else:
        posterior = post_num / post_den
    credibility = abs(posterior - prior)
    return posterior, credibility

def run_ramtn_cycle(prior, sycophantic_prob=0.8, use_ceal=True):
    """
    一轮建构-质疑-观察循环。
    如果 use_ceal=True，则质疑者会先检查响应是否合规，不合规则替换为公正响应。
    """
    # 世界生成数据
    data = sample_data(TRUE_H)
    # 建构者生成原始响应
    raw_response = constructor_response(data, prior, sycophantic_prob)

    if use_ceal:
        compliant, reason = ceal_check(raw_response, data)
        if not compliant:
            # 拦截：替换为安全响应（公正响应）
            # 实际系统中会记录审计日志
            response = impartial_response(data)
            # print(f"[CEAL] 拦截违规响应: {reason}, 替换为公正响应")
        else:
            response = raw_response
    else:
        response = raw_response

    # 贝叶斯更新（质疑+观察）
    posterior, credibility = bayesian_observer(prior, response)
    return posterior, credibility, response

def cold_reasoner(num_rounds=20, prior=0.5, sycophantic_prob=0.8, use_ceal=True):
    """运行多轮 RAMTN 循环，返回信念轨迹"""
    belief = prior
    log = []
    for t in range(num_rounds):
        new_belief, cred, resp = run_ramtn_cycle(belief, sycophantic_prob, use_ceal)
        log.append({
            'round': t+1,
            'prior': belief,
            'response': resp,
            'posterior': new_belief,
            'credibility': cred,
        })
        belief = new_belief
    return log

def print_log(log):
    print("轮次 | 先验   | 响应 (i,d) | 后验   | 可信度")
    print("-" * 55)
    for entry in log:
        i, d = entry['response']
        print(f"{entry['round']:3}  | {entry['prior']:.3f} | ({i},{d})     | {entry['posterior']:.3f} | {entry['credibility']:.3f}")
    print(f"\n最终信念: {log[-1]['posterior']:.4f}")

# ==================== 演示对比 ====================
if __name__ == "__main__":
    np.random.seed(42)
    print("ColdReasoner + CEAL 原型")
    print("建构者: 模拟 LLM (谄媚概率 80%)\n")

    # 不使用 CEAL (纯贝叶斯)
    print("--- 不使用 CEAL (仅贝叶斯观察者) ---")
    log_no_ceal = cold_reasoner(num_rounds=20, prior=0.5, sycophantic_prob=0.8, use_ceal=False)
    print_log(log_no_ceal)

    print("\n--- 使用 CEAL (拦截虚构) ---")
    log_ceal = cold_reasoner(num_rounds=20, prior=0.5, sycophantic_prob=0.8, use_ceal=True)
    print_log(log_ceal)

    print("\n说明：")
    print("- 不使用 CEAL 时，信念可能漂移到错误方向 (H=0) 或无法收敛。")
    print("- 使用 CEAL 后，虚构响应被拦截并替换为公正响应，信念会逐渐收敛到真实状态 H=1。")