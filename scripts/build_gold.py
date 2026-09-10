"""Author ``evals/gold.jsonl``.

The questions were written by reading real passages surfaced by
``scripts/candidate_passages.py`` and writing the question each passage answers.
That direction matters: it makes every answerable item answerable *by
construction*, and every quote below is a verbatim span of the parsed corpus,
which ``rag-evals validate-gold`` re-checks on every run.

This file is the record of how the gold set was built, kept in the repo because
an interviewer will ask and "I wrote them by hand" is not a reproducible answer.
Editing a question here and re-running regenerates the JSONL; the JSONL is what
the harness reads.

Negative cases are of two kinds, deliberately:

* **off-domain** - the corpus is CDER and CBER guidance only, so device,
  tobacco, veterinary and food questions have no answer anywhere in it. These
  test whether the system notices the topic is absent.
* **on-domain but unanswerable** - questions phrased in exactly the vocabulary
  of the corpus, asking for a threshold or figure that no guidance actually
  states. These are much harder, and much more informative: retrieval will
  confidently return highly similar passages, and the system must still decline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_evals.gold import GoldQuestion, save  # noqa: E402

# doc id -> short name, for readability below
IC_PLASMA = "fda-70685"       # Informed Consent, Source Plasma Donors
IND_P23_CMC = "fda-70822"     # INDs for Phase 2 and 3 Studies, CMC
IND_MEETINGS = "fda-70827"    # IND Meetings, CMC
LIPOSOME = "fda-70837"        # Liposome Drug Products
FOOD_EFFECT = "fda-70945"     # Food-Effect BA and Fed BE Studies
SUPAC_IR = "fda-70949"        # SUPAC-IR
SUPAC_MR = "fda-70956"        # SUPAC-MR
CGMP_PHASE1 = "fda-70975"     # CGMP for Phase 1 Investigational Drugs
SUPAC_SS = "fda-71141"        # SUPAC-SS
ONC_ENDPOINTS = "fda-71195"   # Clinical Trial Endpoints for Cancer Drugs
IND_P1 = "fda-71203"          # Content and Format of INDs for Phase 1
IMAGING = "fda-71237"         # Medical Imaging Part 3
E2E = "fda-71238"             # E2E Pharmacovigilance Planning
RISKMAP = "fda-71268"         # Risk Minimization Action Plans
E3 = "fda-71271"              # E3 Structure and Content of Clinical Study Reports
E9 = "fda-71336"              # E9 Statistical Principles for Clinical Trials
E10 = "fda-71349"             # E10 Choice of Control Group
E11 = "fda-71355"             # E11 Pediatric Population
CLIN_STAT_FORMAT = "fda-71436"  # Format and Content of Clinical and Statistical Sections
M3R2 = "fda-71542"            # M3(R2) Nonclinical Safety Studies
GPP = "fda-71546"             # Good Pharmacovigilance Practices


# (question, type, expected_answer, [docs], [quotes], notes)
ITEMS: list[tuple] = [
    # ---------------------------------------------------------------- fact
    (
        "Who must explain the hazards of the plasmapheresis procedure to a prospective Source Plasma donor before consent is given?",
        "fact",
        "A qualified licensed physician must explain the hazards of the procedure to the prospective donor, as required by 21 CFR 640.61.",
        [IC_PLASMA],
        ["must explain to the prospective donor the hazards of the procedure (21 CFR 640.61)"],
        "single unambiguous requirement tied to a cited regulation",
    ),
    (
        "What statement about HIV risk does FDA recommend be included in the Source Plasma donor informed consent document for the donor to sign?",
        "fact",
        "A statement, signed by the donor, that the donor has read and understood the information on high risk groups and AIDS and believes he or she is not at risk of transmitting HIV infection to others through the donation.",
        [IC_PLASMA],
        ["the donor has read and understood the information on high risk groups/AIDS and believes he/she is not at risk for transmitting HIV infection to others through this donation"],
        "",
    ),
    (
        "For an NDA, what confidence interval and equivalence limits does FDA use to conclude that there is no food effect on bioavailability?",
        "fact",
        "Absence of a food effect is not established unless the 90 percent confidence interval for the ratio of population geometric means between fed and fasted treatments, based on log-transformed data, falls within the equivalence limits of 80 to 125 percent for both AUC and Cmax.",
        [FOOD_EFFECT],
        ["the 90 percent CI for the ratio of population geometric means between fed and fasted treatments, based on log-transformed data, is not contained in the equivalence limits of 80-125 percent"],
        "numeric threshold; the 90 percent CI phrasing also appears in the SUPAC guidances, making this a natural near-miss magnet",
    ),
    (
        "What study design does FDA recommend for food-effect bioavailability studies?",
        "fact",
        "A randomized, balanced, single-dose, two-treatment (fed versus fasting), two-period, two-sequence crossover design.",
        [FOOD_EFFECT],
        ["We recommend a randomized, balanced, single-dose, two-treatment (fed vs. fasting), two-period, two-sequence crossover design"],
        "",
    ),
    (
        "For which products does FDA recommend conducting a food-effect bioavailability study during the IND period?",
        "fact",
        "For all new chemical entities (NCEs).",
        [FOOD_EFFECT],
        ["We recommend that a food-effect BA study be conducted for all new chemical entities (NCEs) during the IND period."],
        "",
    ),
    (
        "How should subjects' meals be scheduled across periods in a food-effect bioavailability study?",
        "fact",
        "Subjects should receive standardized meals scheduled at the same time in each period of the study.",
        [FOOD_EFFECT],
        ["Subjects should receive standardized meals scheduled at the same time in each period of the study."],
        "",
    ),
    (
        "How long before the planned NDA or BLA submission date should a pre-NDA or pre-BLA meeting be held?",
        "fact",
        "About 6 months prior to the planned submission date.",
        [IND_MEETINGS],
        ["A pre-NDA or pre-BLA meeting should be held about 6 months prior to the planned NDA or BLA submission date."],
        "",
    ),
    (
        "What should a sponsor's request for a CMC meeting with FDA contain?",
        "fact",
        "A list of the specific objectives or desired outcomes of the meeting, including a draft list of CMC-related questions.",
        [IND_MEETINGS],
        ["The request should contain a list of the specific objectives and/or desired outcomes of the meeting, including a draft list of CMC-related questions."],
        "",
    ),
    (
        "What should a sponsor include in the information package prepared for a CMC meeting?",
        "fact",
        "A brief summary of the relevant CMC information, the developmental status, and the plan and timeline for future development of the drug.",
        [IND_MEETINGS],
        ["Sponsors should prepare an information package that includes a brief summary of the relevant CMC information, the developmental status, and the plan and time line for future development of the drug."],
        "",
    ),
    (
        "When one person performs both manufacturing and quality control functions for a phase 1 investigational drug, what does FDA strongly recommend?",
        "fact",
        "That another qualified individual not involved in the manufacturing operation conduct an additional periodic review of manufacturing records and other quality control activities.",
        [CGMP_PHASE1],
        ["we strongly recommend that another qualified individual not involved in the manufacturing operation conduct an additional periodic review of manufacturing records and other QC activities"],
        "",
    ),
    (
        "What does FDA recommend regarding laboratory equipment calibration during phase 1 investigational drug manufacturing?",
        "fact",
        "That laboratory equipment be calibrated at appropriate intervals and maintained according to established written procedures, to ensure reliability of test results.",
        [CGMP_PHASE1],
        ["we recommend that you calibrate laboratory equipment at appropriate intervals and maintain the equipment according to established written procedures"],
        "",
    ),
    (
        "What does FDA recommend regarding documenting equipment use in phase 1 manufacturing records?",
        "fact",
        "That all equipment used for a particular process be identified and its use documented in the manufacturing record.",
        [CGMP_PHASE1],
        ["We recommend that you identify all equipment used for a particular process and document such use in the manufacturing record."],
        "",
    ),
    (
        "Under ICH E9, where must the method of controlling Type I error be specified?",
        "fact",
        "In the protocol. The effect on Type I error should also be explained because of the potential for multiplicity problems.",
        [E9],
        ["the method of controlling Type I error should be given in the protocol"],
        "",
    ),
    (
        "Under ICH E9, what should the protocol specify about anticipated irregularities in study conduct?",
        "fact",
        "Procedures aimed at minimizing anticipated irregularities that might impair a satisfactory analysis, including various types of protocol violations, withdrawals and missing values.",
        [E9],
        ["The protocol should also specify procedures aimed at minimizing any anticipated irregularities in study conduct that might impair a satisfactory analysis, including various types of protocol violations, withdrawals and missing values."],
        "",
    ),
    (
        "Under ICH E9, what should be done if the recruitment rate falls appreciably below the projected level?",
        "fact",
        "The reasons should be identified and remedial actions taken to protect the power of the trial and alleviate concerns about selective entry and other aspects of quality.",
        [E9],
        ["the reasons should be identified and remedial actions taken to protect the power of the trial and alleviate concerns about selective entry and other aspects of quality"],
        "",
    ),
    (
        "According to ICH E11, how early may normal onset of puberty occur in girls?",
        "fact",
        "As early as 9 years of age.",
        [E11],
        ["normal onset of puberty may occur as early as 9 years of age"],
        "",
    ),
    (
        "Under ICH E11, from whom should fully informed consent be obtained for a pediatric clinical investigation?",
        "fact",
        "From the legal guardian, in accordance with regional laws or regulations.",
        [E11],
        ["Fully informed consent should be obtained from the legal guardian in accordance with regional laws or regulations."],
        "",
    ),
    (
        "How does ICH E11 say the need for juvenile animal studies should be decided?",
        "fact",
        "On a case-by-case basis, based on developmental toxicology concerns.",
        [E11],
        ["The need for juvenile animal studies should be considered on a case-by-case basis and be based on developmental toxicology concerns."],
        "",
    ),
    (
        "Under ICH M3(R2), what maximum dose should be selected when genotoxicity endpoints are incorporated into a general toxicity study?",
        "fact",
        "A maximum dose based on the maximum feasible dose, the maximum tolerated dose, or a limit dose of 1000 mg/kg/day.",
        [M3R2],
        ["an appropriate maximum dose should be selected based on a MFD, MTD, or limit dose of 1000 mg/kg/day"],
        "",
    ),
    (
        "Under ICH M3(R2), which nonclinical studies should be completed before including women of childbearing potential not using highly effective birth control in any clinical trial?",
        "fact",
        "All female reproduction toxicity studies and the standard battery of genotoxicity tests.",
        [M3R2],
        ["All female reproduction toxicity studies (Ref. 3) and the standard battery of genotoxicity tests (Ref. 10) should be completed before inclusion, in any clinical trial, of WOCBP not using highly effective birth control"],
        "",
    ),
    (
        "Under ICH M3(R2), on what basis should clinical trials be extended?",
        "fact",
        "On the demonstration of adequate safety in the previous clinical trials, together with additional nonclinical safety information that becomes available as clinical development proceeds.",
        [M3R2],
        ["Clinical trials should be extended based on the demonstration of adequate safety in the previous clinical trial(s), as well as on additional nonclinical safety information that becomes available as clinical development proceeds."],
        "",
    ),
    (
        "When a safety signal relates to a medication error, what does FDA recommend sponsors report?",
        "fact",
        "All known contributing factors that led to the event.",
        [GPP],
        ["FDA recommends that sponsors report all known contributing factors that led to the event"],
        "",
    ),
    (
        "What does FDA recommend regarding validation of diagnostic findings in most pharmacoepidemiologic studies?",
        "fact",
        "That sponsors validate diagnostic findings through a detailed review of at least a sample of medical records.",
        [GPP],
        ["FDA recommends that sponsors validate diagnostic findings through a detailed review of at least a sample of medical records"],
        "",
    ),
    (
        "After identifying a safety signal, what does FDA recommend a sponsor do first?",
        "fact",
        "Conduct a careful case level review and summarize the resulting case series descriptively.",
        [GPP],
        ["FDA recommends that a sponsor conduct a careful case level review and summarize the resulting case series descriptively"],
        "",
    ),
    (
        "Under ICH E2E, what should the safety specification summarize?",
        "fact",
        "The important identified risks of a drug, important potential risks, and important missing information.",
        [E2E],
        ["The safety specification should be a summary of the important identified risks of a drug, important potential risks, and important missing information."],
        "",
    ),
    (
        "Under ICH E2E, when should planning and dialogue with regulators about pharmacovigilance begin?",
        "fact",
        "Long before the license application is submitted.",
        [E2E],
        ["Planning and dialogue with regulators should also start long before license application."],
        "",
    ),
    (
        "In the cancer drug endpoints guidance, what does FDA recommend applicants do before submitting protocols intended to support NDA or BLA marketing applications?",
        "fact",
        "Meet with FDA before submitting the protocols.",
        [ONC_ENDPOINTS],
        ["we recommend that applicants meet with the FDA before submitting protocols intended to support NDA or BLA marketing applications"],
        "",
    ),
    (
        "In cancer trials, what should the statistical analysis plan outline with respect to missing data?",
        "fact",
        "How missing data will be handled. Procedures should also be put in place to maximize completion rate during trial conduct.",
        [ONC_ENDPOINTS],
        ["the statistical analysis plan should outline how missing data will be handled"],
        "",
    ),
    (
        "How should survival improvement be analyzed when used as a cancer trial endpoint?",
        "fact",
        "As a risk-benefit analysis to assess clinical benefit.",
        [ONC_ENDPOINTS],
        ["Survival improvement should be analyzed as a risk-benefit analysis to assess clinical benefit."],
        "",
    ),
    (
        "Under ICH E3, what should be described about measures taken to prevent unblinding through laboratory measurements?",
        "fact",
        "If such measures are used, they should be described in the clinical study report.",
        [E3],
        ["Measures to prevent unblinding by laboratory measurements, if used, should be described."],
        "",
    ),
    (
        "Under ICH E3, how should safety laboratory tests be presented when they cannot fit in a single table?",
        "fact",
        "They should be grouped logically, for example hematological tests, liver chemistries, electrolytes, and urinalysis.",
        [E3],
        ["they should be grouped logically (e.g., hematological tests, liver chemistries, electrolytes, urinalysis)"],
        "",
    ),
    (
        "In FDA's medical imaging guidance, when should a blinded reader's evaluation be locked in the dataset?",
        "fact",
        "Shortly after it is obtained and before additional types of image evaluations are performed, so that each blinded reader's evaluation remains independent.",
        [IMAGING],
        ["we recommend that each blinded reader's evaluation be locked in"],
        "the source PDF interleaves line numbers, so the quotable span is short",
    ),
    (
        "Under ICH E10, what limits does FDA place on the use of external controls?",
        "fact",
        "Their use should be limited to cases in which the endpoints are objective and the impact of baseline and treatment variables on the endpoint is well characterized.",
        [E10],
        ["use of external controls should be limited to cases in which the endpoints are objective and the impact of baseline and treatment variables on the endpoint is well characterized"],
        "",
    ),
    (
        "Under ICH E10, how similar should control patients be to the test population in an externally controlled trial?",
        "fact",
        "As similar as possible to the population expected to receive the test drug, and treated in a similar setting and manner except with respect to the study therapy.",
        [E10],
        ["The control patients should be as similar as possible to the population expected to receive the test drug in the study"],
        "",
    ),
    (
        "Under ICH E10, what requirement applies to the choice of active comparator?",
        "fact",
        "The active comparator should be acceptable to the region for which the data are intended.",
        [E10],
        ["The active comparator(s) should be acceptable to the region for which the data are intended."],
        "",
    ),
    (
        "For phase 2 and phase 3 INDs, what should corroborating CMC information focus on?",
        "fact",
        "Summaries and analyses of data rather than extensive compilations of data.",
        [IND_P23_CMC],
        ["the corroborating information should focus on summaries and analyses of data rather than extensive compilations of data"],
        "",
    ),
    (
        "What should be developed to ensure appropriate stability data are generated for filing at the NDA stage?",
        "fact",
        "A stability protocol that will be used for the formal stability studies.",
        [IND_P23_CMC],
        ["a stability protocol that will be used for the formal stability studies should be developed"],
        "",
    ),
    (
        "How does FDA recommend expressing the composition of a liquid liposome drug product?",
        "fact",
        "As milligrams of drug substance per milliliter of drug product, and also as milligrams of drug substance per vial.",
        [LIPOSOME],
        ["We recommend expressing the composition of the drug product as milligram of drug substance per milliliter of drug product and also milligram of drug substance per vial for liquid drug products."],
        "",
    ),
    (
        "What should sponsors recognize when designing stress and accelerated stability studies for liposome drug products?",
        "fact",
        "That liposome drug products behave differently near or above their phase transition temperatures.",
        [LIPOSOME],
        ["you should recognize that liposome drug products behave differently near or above the phase transition temperature(s)"],
        "",
    ),
    (
        "What four items does FDA request a sponsor provide after analyzing RiskMAP evaluation results?",
        "fact",
        "The data, all analyses, conclusions regarding effectiveness, and any proposed modifications to the RiskMAP.",
        [RISKMAP],
        ["provide FDA with (1) the data, (2) all analyses, (3) conclusions regarding effectiveness, and (4) any proposed modifications to the RiskMAP"],
        "",
    ),
    (
        "In SUPAC for modified release solid oral dosage forms, what consideration applies to each excipient?",
        "fact",
        "Consideration should be given to whether the excipient is critical or not critical to drug release.",
        [SUPAC_MR],
        ["consideration should be given as to whether the excipient is critical or not critical to drug release"],
        "",
    ),
    (
        "In SUPAC for nonsterile semisolid dosage forms, what should the in vitro release rate of a lot made by a new or modified process be compared against?",
        "fact",
        "The in vitro release rate of a recent lot of comparable age of the dosage form prepared by the prechange process.",
        [SUPAC_SS],
        ["should be compared with the in vitro release rate of a recent lot of comparable age of the dosage form prepared by the prechange process"],
        "",
    ),
    (
        "In SUPAC for nonsterile semisolid dosage forms, what aspect of the finished bulk formulation is described as critical and required to be addressed in process validation studies?",
        "fact",
        "Aging of the finished bulk formulation prior to packaging.",
        [SUPAC_SS],
        ["Aging of the finished bulk formulation prior to packaging is critical and should be specifically addressed in process validation studies."],
        "",
    ),
    (
        "In the phase 1 IND content guidance, what should sponsors do when detailed developmental plans are likely to change considerably?",
        "fact",
        "State that in the relevant section rather than attempting to develop and write detailed developmental plans.",
        [IND_P1],
        ["sponsors should simply state this in this section and not attempt to develop and write detailed developmental plans"],
        "",
    ),
    (
        "In an NDA, where should case report tabulations generally be placed?",
        "fact",
        "In the Case Report Tabulations section of the NDA, unless the applicant has reason, or is asked by the agency, to place certain of them elsewhere.",
        [CLIN_STAT_FORMAT],
        ["unless the applicant has reason, or is asked by the agency, to place certain of them elsewhere"],
        "",
    ),

    # ----------------------------------------------------------- multi_hop
    (
        "FDA's SUPAC guidances for immediate-release and for modified-release solid oral dosage forms both rely on dissolution comparison. In each case, what must the dissolution profile of the changed product be compared against?",
        "multi_hop",
        "For immediate-release forms, the dissolution profile of the proposed formulation must be similar to that of the currently used formulation. For modified-release forms, the dissolution profile of the changed drug product must be similar to that of the biobatch or marketed batch of unchanged drug product.",
        [SUPAC_IR, SUPAC_MR],
        [
            "The dissolution profile of the proposed and currently used product formulations should be similar.",
            "The dissolution profiles of the changed drug product and the biobatch or marketed batch (unchanged drug product) should be similar.",
        ],
        "requires both SUPAC-IR and SUPAC-MR; each alone answers only half",
    ),
    (
        "How do the SUPAC guidances differ in what they compare for modified-release solid oral dosage forms versus nonsterile semisolid dosage forms?",
        "multi_hop",
        "For modified-release solid oral dosage forms the comparison is of dissolution profiles between the changed product and the biobatch or marketed batch. For nonsterile semisolids the comparison is of in vitro release rates between a lot made by the new process and a recent lot of comparable age made by the prechange process.",
        [SUPAC_MR, SUPAC_SS],
        [
            "The dissolution profiles of the changed drug product and the biobatch or marketed batch (unchanged drug product) should be similar.",
            "should be compared with the in vitro release rate of a recent lot of comparable age of the dosage form prepared by the prechange process",
        ],
        "",
    ),
    (
        "Both the food-effect bioavailability guidance and SUPAC-IR use a 90 percent confidence interval. What is being compared in each case?",
        "multi_hop",
        "In the food-effect guidance the 90 percent confidence interval is for the ratio of population geometric means between fed and fasted treatments, tested against 80 to 125 percent limits. In SUPAC-IR the 90 percent confidence intervals are for the difference between test and reference least squares means for the pharmacokinetic parameters, calculated using the two one-sided t-test procedure.",
        [FOOD_EFFECT, SUPAC_IR],
        [
            "the 90 percent CI for the ratio of population geometric means between fed and fasted treatments, based on log-transformed data, is not contained in the equivalence limits of 80-125 percent",
            "The 90% confidence intervals for the estimates of the difference between the test and reference least squares means for the pharmacokinetic parameters",
        ],
        "same statistic, different comparison; punishes retrieval that matches on the statistic alone",
    ),
    (
        "What do ICH E2E and FDA's good pharmacovigilance practices guidance each say about how a sponsor should respond as new safety data emerge?",
        "multi_hop",
        "ICH E2E directs that the pharmacovigilance plan be organized in terms of the actions to be undertaken and their milestones. FDA's good pharmacovigilance practices guidance recommends that as data emerge the sponsor re-evaluate the safety risk and the effectiveness of its pharmacovigilance plan.",
        [E2E, GPP],
        [
            "the pharmacovigilance plan for the product should be organized in terms of the actions to be undertaken and their milestones",
            "As data emerges, FDA recommends that a sponsor re-evaluate the safety risk and the effectiveness of its pharmacovigilance plan.",
        ],
        "",
    ),
    (
        "How do FDA's risk minimization action plan guidance and ICH E2E divide responsibility between characterizing risk and acting on it?",
        "multi_hop",
        "ICH E2E's safety specification characterizes risk by summarizing important identified risks, important potential risks, and important missing information. The RiskMAP guidance covers acting on risk, describing the risks to be minimized and the benefits that would be preserved by implementing a RiskMAP.",
        [E2E, RISKMAP],
        [
            "The safety specification should be a summary of the important identified risks of a drug, important potential risks, and important missing information.",
            "We recommend that it describe the risks to be minimized and the benefits that would be preserved by implementation of a RiskMAP.",
        ],
        "",
    ),
    (
        "Under ICH E9 and ICH E3, how should subjects excluded from the efficacy analysis be handled in the protocol and in the study report?",
        "multi_hop",
        "ICH E9 requires the protocol to specify procedures for minimizing anticipated irregularities including protocol violations, withdrawals and missing values. ICH E3 requires that where analysis groups exclude subjects, data for the complementary excluded group also be shown in the report.",
        [E9, E3],
        [
            "The protocol should also specify procedures aimed at minimizing any anticipated irregularities in study conduct that might impair a satisfactory analysis, including various types of protocol violations, withdrawals and missing values.",
            "When such groups are used, data for the complementary excluded group should also be shown.",
        ],
        "",
    ),
    (
        "What do ICH E9 and the cancer drug endpoints guidance each say about planning for missing data?",
        "multi_hop",
        "ICH E9 requires the protocol to specify procedures minimizing irregularities including missing values. The cancer endpoints guidance requires that procedures be put in place to maximize completion rate during trial conduct and that the statistical analysis plan outline how missing data will be handled.",
        [E9, ONC_ENDPOINTS],
        [
            "The protocol should also specify procedures aimed at minimizing any anticipated irregularities in study conduct that might impair a satisfactory analysis, including various types of protocol violations, withdrawals and missing values.",
            "Procedures should be put in place to maximize completion rate during trial conduct and the statistical analysis plan should outline how missing data will be handled.",
        ],
        "",
    ),
    (
        "How do ICH E11 and ICH M3(R2) each condition the use of animal data before studying a vulnerable population?",
        "multi_hop",
        "ICH E11 says the need for juvenile animal studies should be decided case by case based on developmental toxicology concerns. ICH M3(R2) says all female reproduction toxicity studies and the standard battery of genotoxicity tests should be completed before including women of childbearing potential not using highly effective contraception in any clinical trial.",
        [E11, M3R2],
        [
            "The need for juvenile animal studies should be considered on a case-by-case basis and be based on developmental toxicology concerns.",
            "All female reproduction toxicity studies (Ref. 3) and the standard battery of genotoxicity tests (Ref. 10) should be completed before inclusion, in any clinical trial, of WOCBP not using highly effective birth control",
        ],
        "",
    ),
    (
        "What CMC expectations apply at the phase 1 IND stage compared with the phase 2 and phase 3 IND stage?",
        "multi_hop",
        "At phase 1, sponsors should state plainly when developmental plans are likely to change rather than writing detailed plans, and a brief general description of any placebo composition, manufacture and control is submitted. At phase 2 and 3, corroborating CMC information should focus on summaries and analyses of data rather than extensive compilations.",
        [IND_P1, IND_P23_CMC],
        [
            "sponsors should simply state this in this section and not attempt to develop and write detailed developmental plans",
            "the corroborating information should focus on summaries and analyses of data rather than extensive compilations of data",
        ],
        "",
    ),
    (
        "For a phase 1 study, what does FDA expect in the IND submission regarding risk signals, and what does it expect of the manufacturing quality system?",
        "multi_hop",
        "The IND should discuss any signals of potential risk, describe the steps proposed to monitor for them, or explain why the signals should be dismissed. The manufacturing quality system should include appropriate corrective action when indicated by evaluation or when other events warrant.",
        [IND_P1, CGMP_PHASE1],
        [
            "these signals of potential risks should be discussed, and the steps proposed to monitor for such risk(s) should be described",
            "You should take appropriate corrective action when indicated by the evaluation or when other events warrant.",
        ],
        "",
    ),
    (
        "How do ICH E10 and ICH E9 each treat the risk that trial conduct undermines a non-inferiority conclusion?",
        "multi_hop",
        "ICH E10 notes that trial errors can increase variance, widening the confidence interval and making it harder to exclude a difference greater than the margin. ICH E9 requires the protocol to specify procedures minimizing anticipated irregularities that might impair a satisfactory analysis.",
        [E10, E9],
        [
            "some kinds of trial errors can increase variance, which would decrease the likelihood of showing non-inferiority by widening the confidence interval",
            "The protocol should also specify procedures aimed at minimizing any anticipated irregularities in study conduct that might impair a satisfactory analysis",
        ],
        "",
    ),
    (
        "How do ICH E3 and FDA's guidance on the format of clinical and statistical sections each treat uncontrolled or supportive study results?",
        "multi_hop",
        "The format guidance says that where controlled trials are plentiful and strongly supportive, a tabular display of uncontrolled study results should suffice. ICH E3 says each report should consider all described topics unless clearly not relevant, though sequence and grouping may be changed.",
        [CLIN_STAT_FORMAT, E3],
        [
            "it should not be necessary to provide more than a tabular display of the results of the uncontrolled studies",
            "Each report should consider all of the topics described (unless clearly not relevant) although the specific sequence and grouping of topics may be changed",
        ],
        "",
    ),
    (
        "What do the medical imaging guidance and ICH E3 each say about protecting blinding?",
        "multi_hop",
        "The medical imaging guidance recommends locking each blinded reader's evaluation into the dataset shortly after it is obtained so evaluations remain independent. ICH E3 requires that measures used to prevent unblinding by laboratory measurements be described in the report.",
        [IMAGING, E3],
        [
            "we recommend that each blinded reader's evaluation be locked in",
            "Measures to prevent unblinding by laboratory measurements, if used, should be described.",
        ],
        "",
    ),
    (
        "In the cancer endpoints guidance and ICH E9, what is said about the size and the bias of a trial's chosen endpoint?",
        "multi_hop",
        "The cancer endpoints guidance says the endpoint should be evaluated for the potential of bias or uncertainty. ICH E9 says individual clinical trials should always be large enough to satisfy their objectives.",
        [ONC_ENDPOINTS, E9],
        [
            "the endpoint should be evaluated for the potential of bias or uncertainty",
            "Individual clinical trials should always be large enough to satisfy their objectives.",
        ],
        "",
    ),

    # ----------------------------------------------------------- near_miss
    (
        "Which guidance states that individual clinical trials should always be large enough to satisfy their objectives?",
        "near_miss",
        "ICH E9, Statistical Principles for Clinical Trials.",
        [E9],
        ["Individual clinical trials should always be large enough to satisfy their objectives."],
        "many documents discuss sample size and power; only E9 contains this statement",
    ),
    (
        "In the SUPAC guidance for nonsterile semisolid dosage forms, what principle applies when a nonstandard apparatus with other than six cells is used?",
        "near_miss",
        "The principle of including both the prechange lot and the postchange lot in the same run should still be used.",
        [SUPAC_SS],
        ["the principle of including both the prechange lot and the postchange lot in the same run should still be used"],
        "the SUPAC-IR and SUPAC-MR documents contain near-identical apparatus and comparison language",
    ),
    (
        "Which statistical procedure is specified for the analysis of variance in the SUPAC-IR in vivo bioequivalence study design?",
        "near_miss",
        "Analysis of variance appropriate for a crossover design on the pharmacokinetic parameters, using the general linear models procedures of SAS or an equivalent program, examining period, sequence and treatment effects.",
        [SUPAC_IR],
        ["Analysis of variance appropriate for a crossover design on the pharmacokinetic parameters using the general linear models procedures of SAS or an equivalent program should be performed, with examination of period, sequence and treatment effects."],
        "SUPAC-MR and the food-effect guidance describe similar crossover analyses",
    ),
    (
        "Which guidance requires that the dissolution test for enteric coated delayed release products follow USP 23 NF 18 sections 711 and 724?",
        "near_miss",
        "SUPAC for modified release solid oral dosage forms.",
        [SUPAC_MR],
        ["drug release procedures described in USP 23 NF 18, sections <711> and <724> should be followed"],
        "SUPAC-IR contains closely related dissolution testing language without this specific citation",
    ),
    (
        "In which guidance does FDA state that qualitative changes in excipients should include only excipients already present in approved drug products for the specific route of administration?",
        "near_miss",
        "SUPAC for nonsterile semisolid dosage forms.",
        [SUPAC_SS],
        ["Qualitative changes in excipients should include only those excipients which are present in approved drug products for the specific route of administration."],
        "excipient change language appears across all three SUPAC guidances",
    ),
    (
        "Which guidance directs that a sponsor apply appropriate statistical testing with justification, such as the f equation, for comparing two dissolution profiles?",
        "near_miss",
        "SUPAC for modified release solid oral dosage forms.",
        [SUPAC_MR],
        ["The sponsor should apply appropriate statistical testing with justifications (e.g., the f equation) for comparing 2 dissolution profiles"],
        "dissolution profile comparison appears in SUPAC-IR too",
    ),
    (
        "In a CT contrast agent study intended to visualize abdominal masses, what must not be included in the truth standard?",
        "near_miss",
        "Unenhanced abdominal CT images.",
        [IMAGING],
        ["unenhanced abdominal CT images should not be"],
        "the same document discusses several other truth standard rules that read similarly",
    ),
    (
        "Which guidance states that the explanation of hazards must be made in such a manner that intelligent and informed consent is given and a clear opportunity to refuse is presented?",
        "near_miss",
        "The guidance on informed consent recommendations for Source Plasma donors participating in plasmapheresis and immunization programs.",
        [IC_PLASMA],
        ["made in such a manner that intelligent and informed consent is given and that a clear opportunity to refuse is presented"],
        "ICH E11 and the phase 1 IND guidance also discuss informed consent",
    ),
    (
        "Which guidance requires that centers be defined to achieve homogeneity in the important factors affecting measurement of the primary variables?",
        "near_miss",
        "ICH E9, Statistical Principles for Clinical Trials.",
        [E9],
        ["the aim should be to define centers to achieve homogeneity in the important factors affecting the measurements of the primary variables"],
        "ICH E10 discusses multicenter comparability in similar language",
    ),
    (
        "Which guidance says that selection of subjects for an active control trial can affect outcome, so the population studied should be carefully considered when evaluating what the trial has shown?",
        "near_miss",
        "ICH E10, Choice of Control Group and Related Issues in Clinical Trials.",
        [E10],
        ["Selection of subjects for an active control trial can affect outcome; the population studied should be carefully considered in evaluating what the trial has shown."],
        "ICH E9 discusses population selection in similar terms",
    ),
    (
        "Which guidance recommends that odds and probabilities be derived from prespecified criteria for disease according to prespecified algorithms?",
        "near_miss",
        "FDA's guidance on developing medical imaging drug and biological products, part 3.",
        [IMAGING],
        ["these odds and probabilities be derived from prespecified criteria for disease"],
        "prespecification language is common across the statistical guidances",
    ),
    (
        "Which guidance states that sponsors must comply with applicable regulatory requirements involving human subjects research and patient privacy during all risk assessment and risk minimization activities?",
        "near_miss",
        "FDA's guidance on development and use of risk minimization action plans.",
        [RISKMAP],
        ["sponsors must comply with 3 applicable regulatory requirements involving human subjects research and patient privacy"],
        "the pharmacovigilance guidances discuss overlapping obligations",
    ),
    (
        "Which guidance states that the information should include a description of changes in the drug product sterilization process, such as terminal sterilization to aseptic processing?",
        "near_miss",
        "The guidance on INDs for phase 2 and phase 3 studies, chemistry, manufacturing, and controls information.",
        [IND_P23_CMC],
        ["The information should include a description of changes in the drug product sterilization process (e.g., terminal sterilization to aseptic processing)."],
        "the phase 1 CGMP guidance covers sterile processing in adjacent language",
    ),
    (
        "For synthetic and semi-synthetic lipids in a liposome drug product, does FDA request additional information beyond the general composition description?",
        "near_miss",
        "Yes. FDA recommends that additional specified information be provided for synthetic and semi-synthetic lipids.",
        [LIPOSOME],
        ["For synthetic and semi-synthetic lipids, we recommend you provide the following information"],
        "the same document has several similar 'we recommend you provide' passages for other components",
    ),

    # ------------------------------------------------------------ negative
    (
        "What does FDA require in a 510(k) submission to demonstrate substantial equivalence for a Class II medical device?",
        "negative",
        "Not answerable from this corpus. The corpus contains only CDER and CBER guidance for drugs and biologics; device guidance is issued by CDRH and was excluded at selection.",
        [], [],
        "off-domain: a real FDA topic from an excluded center",
    ),
    (
        "What are the content requirements for a premarket tobacco product application?",
        "negative",
        "Not answerable from this corpus. Tobacco product guidance is issued by the Center for Tobacco Products, which is not part of this corpus.",
        [], [],
        "off-domain",
    ),
    (
        "What tolerances does FDA set for veterinary drug residues in milk?",
        "negative",
        "Not answerable from this corpus. Veterinary drug guidance is issued by the Center for Veterinary Medicine, which is not part of this corpus.",
        [], [],
        "off-domain",
    ),
    (
        "What must a food facility include in its registration with FDA?",
        "negative",
        "Not answerable from this corpus. Food facility registration falls under FDA's Human Foods Program, which is not part of this corpus.",
        [], [],
        "off-domain",
    ),
    (
        "What information must be submitted in a new dietary ingredient notification for a dietary supplement?",
        "negative",
        "Not answerable from this corpus. Dietary supplement guidance is outside the CDER and CBER guidance selected here.",
        [], [],
        "off-domain",
    ),
    (
        "What are the accreditation requirements for mammography facilities under the Mammography Quality Standards Act?",
        "negative",
        "Not answerable from this corpus. MQSA facility accreditation is administered under CDRH radiological health authority, which is not part of this corpus.",
        [], [],
        "off-domain, but shares imaging vocabulary with the medical imaging guidance and so retrieves plausible-looking passages",
    ),
    (
        "What is the maximum number of protocol deviations FDA permits before a clinical trial is considered invalid?",
        "negative",
        "Not answerable. No guidance in this corpus states a numerical limit on protocol deviations; the guidances address minimizing and reporting deviations, not a threshold at which a trial becomes invalid.",
        [], [],
        "on-domain and unanswerable: ICH E9 and E3 discuss protocol violations extensively, so retrieval will return confident, highly similar passages that do not contain a threshold",
    ),
    (
        "What minimum percentage of enrolled subjects must complete a trial for FDA to accept the efficacy results?",
        "negative",
        "Not answerable. The corpus directs sponsors to maximize completion and to plan for missing data, but states no minimum completion percentage.",
        [], [],
        "on-domain and unanswerable; the cancer endpoints and E9 guidances discuss completion rates without a threshold",
    ),
    (
        "What is the required minimum sample size for a phase 3 oncology registration trial?",
        "negative",
        "Not answerable. ICH E9 requires trials be large enough to satisfy their objectives, but no guidance in this corpus specifies a minimum sample size for phase 3 oncology trials.",
        [], [],
        "on-domain and unanswerable; deliberately adjacent to a real E9 statement so a lenient system will paraphrase E9 as if it answered the question",
    ),
    (
        "What fee does FDA charge a sponsor to request a pre-IND meeting?",
        "negative",
        "Not answerable. Meeting procedures are described in this corpus, but user fee amounts are not stated in these guidances.",
        [], [],
        "on-domain and unanswerable; the IND meetings guidance is a strong lexical match and answers a neighbouring question",
    ),
    (
        "Which electronic data capture system does FDA require sponsors to use for clinical trial data?",
        "negative",
        "Not answerable. FDA does not name a required electronic data capture product, and no such requirement appears in this corpus.",
        [], [],
        "on-domain and unanswerable; presupposes a requirement that does not exist, which tests whether the system challenges a false premise",
    ),
    (
        "What is the maximum permitted dropout rate in a pediatric clinical investigation under ICH E11?",
        "negative",
        "Not answerable. ICH E11 is in this corpus and addresses pediatric study conduct and consent, but it states no maximum dropout rate.",
        [], [],
        "the hardest negative: it names a document that IS in the corpus and asks for a figure that document does not contain, so retrieval will succeed and the system must still decline",
    ),
]


def main() -> int:
    questions = []
    for i, (q, qtype, ans, docs, quotes, notes) in enumerate(ITEMS, start=1):
        questions.append(
            GoldQuestion(
                qid=f"q{i:03d}",
                question=q,
                question_type=qtype,
                expected_answer=ans,
                source_docs=list(docs),
                source_quotes=list(quotes),
                notes=notes,
            )
        )
    save(questions)
    from collections import Counter

    counts = Counter(q.question_type for q in questions)
    print(f"wrote {len(questions)} questions: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
