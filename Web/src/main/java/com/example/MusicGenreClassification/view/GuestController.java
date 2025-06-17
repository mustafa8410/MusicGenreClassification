package com.example.MusicGenreClassification.view;

import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;

import java.util.UUID;

@Controller
public class GuestController {
    @PostMapping("/guest/start")
    public String startGuestSession(HttpSession session) {
        if (session.getAttribute("guestId") == null) {
            session.setAttribute("guestId", UUID.randomUUID().toString());
        }
        // Optionally set a flag
        session.setAttribute("isGuest", true);
        return "redirect:/home";
    }
}


