package com.example.MusicGenreClassification.view;

import jakarta.servlet.http.HttpSession;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.security.Principal;

@Controller
public class HomeController {
    @GetMapping("/home")
    public String homePage(HttpSession session, Model model, Principal principal) {
        // If authenticated, show user details
        if (principal != null) {
            model.addAttribute("username", principal.getName());
            model.addAttribute("isGuest", false);
        } else if (session.getAttribute("guestId") != null) {
            model.addAttribute("isGuest", true);
        } else {
            model.addAttribute("isGuest", false);
        }
        return "home";
    }
}

