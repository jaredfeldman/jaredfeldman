from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_binary
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException


# Get analysis function
def analyze_game(game_id):
    # Initializes chrome web driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.set_page_load_timeout(15)

    try:
        # Construct the game URL
        game_url = f"https://lichess.org/{game_id}"
        driver.get(game_url)

        # Find the "Sign in" link and click it
        sign_in_link = driver.find_element(By.XPATH, "//a[contains(@href, '/login?referrer=') and contains(@class, 'signin')]")
        sign_in_link.click()

        # Find the username input field and enter your username
        username_input = driver.find_element(By.ID, "form3-username")
        username = "BCU_555"  # Enter your username
        username_input.send_keys(username)

        # Find the password input field and enter your password
        password_input = driver.find_element(By.ID, "form3-password")
        password = "(s,XkGvJu,A@c5:"  # Enter your password
        password_input.send_keys(password)

        # Submit the login form
        login_button = driver.find_element(By.XPATH, "//button[contains(., 'Sign in')]")
        login_button.click()

        # Wait for the page to load after login
        print("     waiting for page to load. 3s")
        wait = WebDriverWait(driver, 3000)  # Adjust the timeout as needed

        # Check if the "Computer analysis" tab is present
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//span[@role='tab' and text()='Computer analysis']")))
            print("     Computer analysis tab is present")
        except TimeoutException:
            print("     Computer analysis tab not found. Exiting.")
            return
            
        computer_analysis_tab = driver.find_element(By.XPATH, "//span[@role='tab' and text()='Computer analysis']")

        # Switch to the computer analysis tab
        computer_analysis_tab.click()
        print("     Computer analysis tab clicked")

        # Need to locate the computer button  first
        # Check if the "Computer analysis" button is present
        try:
            # Locate and click the "Request a computer analysis" button
            button = driver.find_element(By.XPATH, "//button[@type='submit' and contains(., 'Request a computer analysis')]")
            button.click()
            print("     Computer analysis button clicked")

        except NoSuchElementException:
            print("Request a computer analysis button not found. Exiting.")
            return

    finally:
        # Close the browser after analysis
        driver.quit()