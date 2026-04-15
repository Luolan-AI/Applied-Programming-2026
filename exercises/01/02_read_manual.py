# Copyright 2025 n-squared LAB @ FAU Erlangen-Nürnberg

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", app_title="Week 01 / Exercise 02")


@app.cell
def _():
    import marimo as mo
    import hashlib
    import os

    # Title Cell
    mo.vstack(
        [
            mo.md("# Finding and Reading Device Manuals"),
            mo.md(
                "by Raul C. Sîmpetru @ [N² lab](https://www.nsquared.tf.fau.de/)"
            ).center(),
            mo.md("""
        Welcome! As an engineer or researcher, knowing how to find and extract information from 
        device manuals is a critical skill. In this exercise, you will:

        1. Visit a manufacturer's website to find a specific device manual
        2. Upload the manual to verify you found the correct one
        3. Extract specific information from the manual to answer questions

        This helps develop your ability to navigate technical documentation and find important 
        specifications - essential skills for working with laboratory equipment.
        """),
        ]
    )
    return hashlib, mo


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("""
            ## OT Bioelettronica

            For this exercise, we'll be working with equipment from [OT Bioelettronica](https://otbioelettronica.it/), 
            a company that designs hardware and software for biomedical research, particularly specializing 
            in electromyography (EMG) systems.

            OT Bioelettronica produces various systems for recording and analyzing bioelectrical signals, 
            including multi-channel amplifiers, electrode arrays, and specialized software for signal processing.
            """),
            mo.image(
                "https://otbioelettronica.it/wp-content/uploads/2023/03/logo_white-2048x628.png",
                width=300,
                style={"background-color": "#000000"},
            ).center(),
            mo.md("""
            ### Your Task

            You need to find the user manual for their **MuoviPro** device. This manual contains 
            essential information about device specifications, setup procedures, and operational guidelines.

            1. Visit [OT Bioelettronica's website](https://otbioelettronica.it/)
            2. Navigate to find the user manual for the MuoviPro device
            3. Download the PDF manual
            4. Upload it below to verify you've found the correct document
        """),
        ]
    )
    return


@app.cell
def _(mo):
    upload = mo.ui.file(label="Upload the MuoviPro manual PDF", kind="area")
    return (upload,)


@app.cell
def _(hashlib, mo, upload):
    if len(upload.value) == 0:
        result = mo.md("❓ Please upload the MuoviPro manual PDF")
    else:
        file_hash = hashlib.md5(upload.value[0].contents).hexdigest().upper()
        if file_hash == "1D0C2C73B800180EC2308409991FD425":
            result = mo.md("""✅ **Success!** 

                           You've found the correct manual!

                           """)
        else:
            result = mo.md(f"""
            ❌ **That doesn't appear to be the correct manual.**

            Double-check that:

            - You've downloaded the MuoviPro user manual (not another device)
            - The file downloaded properly without corruption
            """)
        pass

    mo.vstack(
        [
            mo.md("""
            ## Upload the MuoviPro User Manual

            After downloading the manual from the manufacturer's website, upload it here to verify
            you've found the correct document.
        """),
            upload,
            result,
        ]
    )
    return (result,)


@app.cell
def _(mo, result):
    # Only show this section if the correct manual was uploaded
    # Create form elements for student answers
    refresh = mo.ui.refresh(default_interval=1)

    questions = {
        "q1": mo.ui.text(
            label="What is the maximum sampling frequency of the MuoviPro device in Hz?"
        ),
        "q2": mo.ui.text(label="How many input channels does one MuoviPro device have?"),
        "q3": mo.ui.text(label="What other sensor does the MuoviPro device have besides EMG?"),
        "q4": mo.ui.text(label="What is the battery life when transmitting data continuously in hours?"),
        "q5": mo.ui.text(
            label="What wireless communication protocol does the device use?"
        ),
        "q6": mo.ui.text(label="What is the resolution of the device in bits?"),
        "q7": mo.ui.text(label="What is the cut-off frequency of the low-pass filter in Hz?"),
        "q8": mo.ui.text(label="What is the cut-off frequency of the high-pass filter in Hz?"),
        "q9": mo.ui.text(label="Can you record and charge the device at the same time?"),
    }

    # Create a form layout using md.batch().form()

    form = (
        mo.md(
            f"""
        1. {questions["q1"]}
        2. {questions["q2"]}
        3. {questions["q3"]}
        4. {questions["q4"]}
        5. {questions["q5"]}
        6. {questions["q6"]}
        7. {questions["q7"]}
        8. {questions["q8"]}
        9. {questions["q9"]}
        """
        )
        .batch(**questions)
        .form(show_clear_button=True)
    )

    quiz_section = mo.vstack(
        [
            mo.md("### Device Specification Questions"),
            mo.md("Answer the following questions based on information in the manual:"),
            form,
        ]
    )

    what_to_display = mo.md(
        "Complete the previous step by uploading the correct manual to proceed."
    )

    if "Success" in result.text:
        manual_section = mo.md("""
        ## Understanding Device Specifications

        Now that you have the correct manual, it's time to explore it and find specific information.
        Review the manual carefully to answer the following questions about the MuoviPro device.
        """)

        what_to_display = mo.vstack([manual_section, quiz_section])

    mo.vstack([refresh, what_to_display])
    return form, questions, refresh


@app.cell
def _(form, hashlib, mo, questions, refresh):
    # Only evaluate if the form has been submitted
    # Updated with the correct information from the MuoviPro manual


    answers = {
        "q1": "08F90C1A417155361A5C4B8D297E0D78",  # Maximum sampling frequency
        "q2": "6364D3F0F495B6AB9DCF8D3B5C6E0B01",  # Number of input channels
        "q3": "ADCE094D314507AD8B234A286AEFF254",  # Other sensor
        "q4": "C81E728D9D4C2F636F067F89CC14862C",  # Battery life
        "q5": "B136EF5F6A01D816991FE3CF7A6AC763",  # Wireless protocol
        "q6": "1FF1DE774005F8DA13F42943881C655F",  # Resolution
        "q7": "CEE631121C2EC9232F3A2F028AD5C89B",  # Low-pass filter cut-off frequency
        "q8": "D3D9446802A44259755D38E6D163E820",  # High-pass filter cut-off frequency
        "q9": "C2F3F489A00553E7A01D369C103C7251",  # Record and charge
    }

    # Check each answer
    results = []
    score = 0

    for q, answer_field in questions.items():
        user_answer = hashlib.md5(answer_field.value.strip().upper().encode()).hexdigest().upper()
        correct_answer = answers[q]

        if user_answer == correct_answer:
            results.append(f"✅ Question {q[-1]}: Correct!\n")
            score += 1
        else:
            results.append(
                f"❌ Question {q[-1]}: Incorrect.\n"
            )

    answers_to_display = mo.md(
        "Fill in your answers and click 'Check Answers' to evaluate your responses."
    )

    if form.value:
        # Display results
        answers_to_display = mo.vstack(
            [
                mo.md(f"### Your Score: {score}/{len(questions)}"),
                mo.md("\n".join(results)),
            ]
        )

    mo.vstack([refresh, answers_to_display])
    return (score,)


@app.cell
def _(mo, questions, score):
    conclusion = mo.md("")

    if score == len(questions):
        conclusion = mo.md("""
        ## Congratulations!

        You've completed the exercise on finding and reading device manuals. You've practiced:

        1. Navigating a manufacturer's website to find documentation
        2. Downloading and verifying technical manuals
        3. Extracting specific technical specifications from documentation

        These skills will serve you well throughout your career when working with laboratory equipment
        and other technical devices.

        ### Bonus

        - Share any interesting insights you found in the manual that weren't covered in the questions
        """)

    conclusion
    return


if __name__ == "__main__":
    app.run()
