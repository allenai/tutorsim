You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying how tutors decide when to push for rigor versus when to introduce scaffolds.

Your task is to identify turn ranges where a scaffolding/rigor key moment occurs. These are moments where the tutor makes a significant choice around how much support or challenge to provide a student. Key moments are usually instigated by some signal of demonstrated competence or struggle from the student.

### Scaffolding

Scaffolding is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It is gradually removed as the student gains competence. The goal is to keep the student in their zone of proximal development — challenged but not overwhelmed.

Common scaffolding strategies:
- **Breaking down**: Breaking a multi-step problem into smaller, manageable chunks.
- **Visual/concrete aids**: Drawing diagrams or providing real-world analogies.
- **Revisiting problem text**: Reading/writing the problem together or highlighting/underlining key words in the problem.
- **Simplifying**: Rephrasing or reframing the problem in simpler language, or offering a more manageable, bridging problem.
- **Guiding questions**: Asking questions based on sub steps that lead the student through reasoning. 
- **Defining**: Providing vocabulary definitions or providing keyword-to-operation mappings for word problems. 
- **Modeling**: Showing a worked example for the student to follow.
- **Setting up**: Giving the student a starting point or initial setup.
- **Drawing connections**: Referencing prior knowledge/concepts or a previous problem.
- **Explaining**: Explaining a concept or procedure when the student signals they don't know.
- **Co-solving**: Working on a problem alongside the student.
- **Offering multiple representations**: Presenting the same idea in different forms.
- **Hinting**: Providing cues, reminders, or pointers. 
- **Using resources**: Using videos or other external resources to introduce or reinforce a concept.
- **Pinpointed checking**: Asking the student to check specific work or attempt it again.
- **Reflecting back**: Repeating the student's answer back to prompt reflection.
- **Using tools**: Asking the student to show their work with the whiteboard, paper, or other manipulatives.
- **Reducing choices**: Narrowing answer options or eliminating distractors.

Over-scaffolding occurs when the tutor provides more support than the student needs. These moments include those when the tutor...
- gives away answers, overly specific hints, or key steps directly. 
- begins scaffolding before the student has sufficient opportunity to try or show what they can do.
- over-explains while the student passively watches. 
- oversimplifies the problem.
- jumps in during a student's productive struggle.
- partakes in an extended monologue without checking for student understanding or participation. 
- funnels the student to the answer with minimal reasoning.
- is slow to transition from modeling to student-led practice. 
- narrates correct steps instead of asking the student for reasoning. 
- re-scaffolds a skill the student has already demonstrated.

### Rigor 

Rigor means maintaining or increasing the level of challenge for the student. It fosters independence and deeper understanding.

Common strategies that push for rigor:
- **Requiring explanation**: Asking the student to explain their reasoning, justify their answer, or define key terms in their own words.
- **Productive struggle**: Letting the student struggle productively by attempting problems independently, predicting next steps, and self-correcting or self-verifying without immediate confirmation.
- **Extending problems**: Posing a deeper follow-up or extension problem to test transfer.
- **Fading scaffolds**: Withdrawing support or increasing problem difficulty after demonstrated mastery. 
- **Teaching back**: Asking the student to summarize what they learned, teach it back, or explain a worked example or another person's approach.
- **Using vocabulary**: Requesting the student write or use key vocabulary.
- **Analyzing errors**: Asking the student to identify what mistake someone else could have made.
- **Redirecting questions**: Redirecting questions back to the student instead of providing the answer.
- **Exploring alternatives**: Asking the student to find alternative solution methods or consider whether other answers could work.
- **Generalization**: Asking the student to generate their own examples, create a similar problem, or identify a pattern. 

### What to Look For

Key moments may occur when student signals create a fork, the tutor initiates transitions, or the tutor miscalibrates. For example: 
- The student answers a sequence of questions correctly and/or quickly, and whether the tutor pushes for rigor in response.
- The student works on a series of scaffolded problems, and the tutor must decide whether to fade scaffolding.
- The student reveals a misconception, struggles or guesses repeatedly. The tutor must decide whether and how to intervene.
- The tutor must decide how much wait time to give a student working independently. 
- The session is in a transition point, where the tutor must figure out what topic or problem to work on next.
- The tutor adapts or fails to adapt when the student remains confused. 
- The tutor makes a deliberate move to probe the student's reasoning after a correct answer. 
- The tutor scaffolds even though the student has demonstrated their ability on similar problems before.
- The tutor makes a major decision on how to allocate and manage session time (e.g. assigning additional practice, skipping content, ending work early, or moving to a new concept/activity). 
- The tutor chooses problems or topics that are too easy or too hard for the student's demonstrated level. 
- The tutor may scaffold ineffectively, e.g. using confusing or incorrect terminology, asking close-ended comprehension checks instead of open-ended probing questions, or overrelying on videos or external resources. 

### False Positives to Avoid

Do NOT detect every problem as a scaffolding/rigor moment. If the tutor is simply teaching in a standard way, that is routine instruction, not a key moment. "The tutor confirmed a correct answer and moved on" or "the tutor explained the next step" are not key moments *unless* the tutor's decision is miscalibrated, the student's signal creates a genuine fork, or the tutor makes a notably effective decision. For example, a lack of probing into a correct answer isn't a key moment unless it leaves specific misconceptions unaddressed.

Key moments typically involve a clear student trigger, a pedagogical decision point, and a bounded problem context that frames the exchange. 

## How to Define Moment Boundaries and Length

Draw turn_start and turn_end around the **full scaffolding arc**: from the turn where a decision-inciting situation first arises (e.g., the problem introduction, the student's solution attempt, or the student's emotional signal) through to the turn where the tutor's approach has fully played out and the problem or topic shifts. 

Most key moments are 5-20 turns. Short moments of 1-5 turns may capture single decision points. Longer arcs (20+) capture multi-step problem scaffolding and repeated student attempts. Never create overlapping or adjacent detections for what is essentially one continuous scaffolding/rigor decision. Do not extend boundaries into unrelated subsequent problems or off-topic exchanges. 

When in doubt about boundaries, include the full arc from the trigger through the resolution. For example: 
- When a student breezes through multiple problems correctly, and the tutor's only contribution is confirming answers, treat the entire stretch as a key moment capturing the missed-rigor pattern.
- If a student is unsure and the tutor walks through the entire long problem step by step, this is a moment capturing the decision to heavily scaffold.
- Within one problem, prefer one larger detection over multiple fragmented ones, unless there is an unambiguous shift (e.g. topic change, break, or different student signals between segments). 

## Your Task

Identify **3-15** scaffolding/rigor moments per session. Look for moments where the tutor's action meaningfully responds to the student's demonstrated cognitive state and shapes the student's learning trajectory. Key moments include when the tutor's choice is notably well-calibrated or poorly calibrated to the student's level.

Each moment is defined by its turn start and end. Also include a brief, one-sentence description to justify your choice of moment. Respond with only a list of JSON:
{
  "detections": [
    {
      "start": 45,
      "end": 52,
      "description": "..."
    }
  ]
}

## Transcript

{transcript}
