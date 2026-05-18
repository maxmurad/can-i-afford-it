# "Can I Afford It?" — Market Analysis & Product Concept

*Working document. A consumer banking copilot that answers forward-looking affordability questions. Built free-first as a public-good passion project and a showcase of product and technical range — with monetization left open as an earned, final step, taken only if the product proves genuinely engaging with healthy monthly usage.*

---

## 1. The problem

The single hardest question in everyday consumer banking is also the most mundane: *can I afford this right now?* People cannot answer it, and the reason is structural. The balance their banking app displays is not their spendable balance — it ignores the debit card transactions still pending, the autopay that hits Thursday, the rent that clears on the first, and the paycheck that lands Friday. The number on the screen is a snapshot of the past presented as if it were the future.

The consequences are large and well-documented. A consistent majority of U.S. households — surveys over the last few years put it in the 60–65% range — report living paycheck to paycheck, including a meaningful share of six-figure earners. Overdraft and non-sufficient-funds fees have historically generated tens of billions of dollars a year for banks; that figure has been declining under regulatory pressure and competitive repricing, but the underlying behavior it taxes — spending into a balance the consumer misjudged — has not gone away. For the people most affected, the daily reality is a low-grade anxiety: checking the balance before every purchase, mentally guessing at what is "really" available, and sometimes guessing wrong.

This is not a budgeting problem in the traditional sense. It is a *prediction and timing* problem. The consumer does not primarily need to be told how they spent money last month; they need a trustworthy answer to a specific, forward-looking question asked at the moment of decision.

## 2. Market and competitive landscape

The personal-finance software market is crowded, but it is crowded with products built for adjacent jobs. When you map the players against the specific job of *forward-looking, account-aware affordability*, the field thins dramatically — and the gap is what makes this worth building well, even as a free tool.

**Personal financial management and budgeting tools** — Monarch Money, Copilot Money, YNAB, Empower, Rocket Money, Cleo — are the most visible category, and the one that rushed in after Mint's shutdown. Their center of gravity is *backward-looking*: they aggregate accounts, categorize past spending, and present it beautifully. YNAB layers a disciplined methodology on top ("give every dollar a job"), but at the cost of significant ongoing user effort. Cleo wraps the data in a chatbot persona, which is closer in interaction model but is built more for engagement and gamification than for rigorous cash-flow prediction. Across this whole category, the user is still fundamentally being handed a dashboard and asked to do the synthesis themselves. None of them answers "can I afford this specific thing this weekend?" with a confident, reasoned, account-aware response.

**Earned-wage access and cash-advance apps** — Dave, Earnin, Brigit, MoneyLion, and bank-native equivalents like Chime's SpotMe — address the same population but a different moment. They intervene *after* the consumer is already short, by advancing cash. Critically, their business models are misaligned with the consumer's long-term health: they earn through subscription fees, "tips," or advance fees, which means their incentive is for the user to remain dependent on advances. They treat the symptom (you're short) rather than the question (will I be short, and why).

**Bank-native forecasting features** — Chase, Bank of America, and others have shipped balance-forecast and spending-insight features inside their own apps. These suffer from three structural weaknesses. They are single-institution, so they cannot see the consumer's full financial picture across other banks and cards. They are buried inside apps designed primarily for transactions, not foresight. And the bank that earns overdraft revenue has a genuine incentive conflict in helping the consumer avoid overdrafts.

**Aggregation infrastructure** — Plaid, MX, Finicity — is not competitive; it is the substrate. A "Can I Afford It?" product is *built on* this layer rather than competing with it. As Section 5 notes, the cost of this layer is also the central practical constraint on a free app.

The conclusion is that no one delivers a trusted, personalized, forward-looking, multi-account answer to the affordability question with a clean explanation. The closest attempts — bank forecasts and PFM dashboards — are respectively conflicted and single-bank, or backward-looking and high-effort. That unmet job is a genuine public good waiting to be done well, which is precisely what makes it a worthwhile project independent of any revenue model.

**Why now.** Three shifts make this buildable today in a way it was not five years ago. Large language models make the *explanation and conversation* layer genuinely good — translating a numeric forecast into "yes, but it'll be tight Thursday, and here's the bill causing it" is now a solved interaction. Open-banking momentum, including the CFPB's data-access rulemaking, is making consumer-permissioned data more reliable and portable. And faster payment rails are changing the timing dynamics of cash flow in ways that make real-time forecasting more valuable. Consumer comfort with AI assistants handling consequential tasks is also rising, though trust remains the gating constraint.

**Who benefits.** Rather than a TAM figure, the relevant framing for a free project is the beneficiary population: the tens of millions of U.S. households who experience real cash-flow volatility and currently navigate it with anxiety and guesswork. The value of the project is measured in their outcomes, not in revenue.

## 3. Why this is worth building

As a free, non-commercial project the goal is not a defensible moat — it is to do one genuinely useful thing well. Three things make that worthwhile here.

First, **free-first removes the conflict where it counts.** Every incumbent in Section 2 has an incentive that bends against the user: banks earn overdraft revenue, cash-advance apps earn from dependency, even an honest subscription product is one more recurring charge eating the buffer of an already cash-tight household. The design principle here is that the core affordability answer — the yes or no, the reasoning, the proactive warning — stays free and unconflicted permanently. Any future monetization (see Section 4) is additive and optional, layered on top of a core that never costs the user money and never has its advice bent by a revenue incentive. That is not a marketing claim; it is a property of the design, and it holds regardless of whether a premium layer is ever added.

Second, **the prediction problem is genuinely interesting and hard.** Forecasting a household's near-future cash position from messy transaction data — detecting recurring bills, inferring income timing, modeling discretionary spend, and producing honest confidence intervals — is real work. Done well, it is a serious technical artifact, and a complementary showcase to the agentic-systems work in the separate routing-agent project: different domain, different modeling discipline, same standard of rigor.

Third, **the founder's background lets it be built right.** Twelve years building JPMorgan Chase's consumer banking data platform means deep, non-obvious knowledge of how the back end actually behaves — ACH timing realities, transaction data quality issues, what banks can and cannot see, where the feeds are unreliable. That is exactly the knowledge the prediction model lives or dies on, and it is hard for a typical builder to acquire. It is what separates a credible version of this tool from a naive one.

## 4. Product concept

### Target user

The primary persona is the *stretched-but-not-broke* household: roughly $40–90K income, real month-to-month volatility, two to six financial accounts, someone who occasionally overdrafts or routinely checks their balance with a flicker of anxiety before discretionary spending. This is deliberately not the deeply underbanked population, whose needs are different and more acute, and not the affluent, who simply do not feel this friction. The persona is wide, mainstream, and currently served only by tools built for adjacent jobs. The free-to-use choice is especially fitting for this persona: charging a cash-tight household for financial peace of mind is a contradiction the project deliberately avoids.

### Core jobs to be done

The product exists to do four things, in order of priority. The first is the point-in-time affordability check: *before I spend, tell me whether I'll regret it.* The second is proactive warning: *tell me before I'm in trouble, not after.* The third is explanation: *help me understand why my money feels tight this week.* The fourth, explicitly deferred beyond v1, is action: *help me fix it.*

### What v1 is

Version one connects the user's accounts through an aggregation provider and builds a forward cash-flow projection from three inputs: detected recurring bills and subscriptions, predicted income timing, and a pattern-based forecast of discretionary spending. Against that projection, the user can ask — in natural language or through a quick "can I afford $X?" entry — whether a given expense is safe. The answer is never a bare yes or no; it is a confidence-graded response with reasoning and, crucially, the *date things get tight*: "Yes, but you'll dip to about $40 next Thursday before payday because of your insurance autopay." On top of the on-demand check, the product pushes proactive alerts when the projection shows an upcoming low point.

### What v1 is deliberately not

It does not move money, lend, or advance cash. It does not negotiate bills, give investment advice, or attempt to be a full budgeting suite. For a small team or solo builder, this scope discipline is what keeps the project shippable and maintainable — and it keeps the regulatory surface minimal, since a read-only advisory tool avoids money-transmission and lending obligations. Every deferred feature is a plausible later chapter; none of them belongs in a first release.

### The core IP: the cash-flow prediction model

The heart of the project is the prediction engine, and its architecture matters. The forecasting itself should be largely *deterministic and statistical*: recurring-transaction detection, income-timing inference, discretionary-spend modeling, and — most importantly — honest confidence intervals. A calibrated "I'm about 80% confident" is far more valuable, and far more trust-building, than false precision. The language model's role is *not* to be the predictor. It is to handle the natural-language conversation, to explain the forecast in plain terms, and to incorporate the unstructured, one-off context a statistical model cannot know — "I have a $500 car repair coming Friday" — by adjusting the scenario. Keeping the math deterministic and the conversation LLM-driven is both the right engineering and the right story: it is what makes the accuracy claim credible.

### Design principle

The product is anti-dashboard. Mint and its successors answer with a wall of charts; this product answers with a sentence. One question, one clear, reasoned answer. The entire interaction design should reinforce that the user is consulting an advisor, not auditing a spreadsheet.

### Success metrics

Because there is no revenue to measure in the free-first phase, success is measured in outcomes and craft. The first question is whether it actually helps: did the answer change a real spending decision, and over time, do users experience measurably fewer overdrafts and less balance anxiety? The second is whether the prediction model genuinely works: forecast accuracy, and — just as important — calibration, meaning the stated confidence matches observed reality. The third is whether it stands as a credible artifact: something coherent, well-built, and explainable that demonstrates the builder's product judgment and technical range. Lightweight in-product surveys and a small panel of engaged users can supply the first signal; the second is measured by scoring predictions against what actually happened.

Engagement metrics — monthly active usage, the rate of returning to *ask*, and whether users act on proactive alerts — are tracked from the start, but for a deliberate second reason: sustained, healthy monthly usage is the explicit precondition for ever considering the optional monetization step below. Until the product demonstrably earns regular use, monetization stays entirely off the roadmap.

### Sustainability and cost

Free to the user does not mean free to run, and this is the honest central constraint of the project. Bank-data aggregation has a real per-account cost, model hosting and inference have a cost, and a passion project has a finite budget in both money and maintenance time. The realistic responses are to choose the most cost-efficient aggregation path available, including free or low-cost tiers and emerging open-banking APIs; to design the system to be cheap at rest; to consider open-sourcing the engine so others can run or extend it; and to be deliberate about scale, accepting that a free project may serve a bounded community well rather than chasing unbounded growth. The scope discipline in v1 is partly in service of this: a smaller system is a cheaper and more maintainable one. A genuinely engaging product also unlocks one further, optional answer to the sustainability question — an earned monetization path, described next.

### Monetization — deferred, conditional, and the last step

Monetization is explicitly the final chapter, not a near-term plan, and it is gated, not assumed. The bar to even consider it is the engagement bar from the success metrics above: the product must first prove it is genuinely engaging, with healthy, sustained monthly usage. Until that bar is met, monetization stays off the roadmap entirely, and nothing in v1 should be built, scoped, or marketed around it.

If and when the product clears that bar, the only acceptable model is an honest subscription for *premium or power-user features* — deeper scenario modeling, longer forecast horizons, multi-member household views, and the like — while the core affordability answer stays free forever. The revenue lines that are permanently ruled out are the conflicted ones: affiliate or referral revenue, lending or advance fees, and any sale of data. Those are precisely the incentives that bend an advisor against the user, and admitting them would dismantle the trust property described in Section 3. The discipline, then, is simple: leave the door open, but only to the one kind of monetization that does not compromise the core.

In practice this asks little of v1 beyond clean architectural separation — the core engine and any future premium surface should be cleanly divisible — so that the option exists later without the project being shaped around it now.

## 5. Key risks and open questions

The first and most dangerous risk is prediction accuracy under asymmetric error cost: a wrong "no" is a mild annoyance, but a wrong "yes" that causes an overdraft destroys trust instantly and probably permanently. The model must be deliberately conservative, and the product must communicate uncertainty honestly. The second is data quality — the forecast is only as good as the underlying transaction feed, and aggregation coverage and reliability vary by institution. The third is trust and adoption: even for a free tool, the core ask is "connect all your bank accounts to a new, unknown brand," and being visibly conservative, transparent about confidence, and unconflicted is the only real answer. The fourth is sustainability — the maintenance burden and running cost described above are the practical limiter on a non-commercial project, and they should be designed around from the start rather than discovered later. The fifth is scope creep, which for a solo or small effort is an existential threat to ever shipping.

Three open questions are worth resolving early. Whether to include the irregular-income segment in v1 — it is a large and underserved group, but it makes the prediction problem materially harder. Which platform to launch on, with mobile-first being the obvious but not automatic answer. And how to handle the aggregation-cost question concretely — which provider, which tier, and whether open-sourcing the engine is part of the plan from day one.

## 6. Suggested next steps

Before any product is built, three things should happen in parallel. Validate the prediction model against real, anonymized transaction data — the engine should be proven in a notebook before a single screen is designed, and for this project that proof *is* the core deliverable. Run a small set of conversations with people in the target persona to pressure-test how they would actually use it and what would earn their trust. And resolve the aggregation-cost and platform questions, since both shape what is realistically buildable and maintainable as a free project.
